#pragma once
#include <functional>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/math.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl;
using fp16 = sycl::half;

// ============================================================================
// Prefill FMHA — kBr=8, kBc=16, DPAS score, scalar max (avoid reduce bug),
// SIMD exp, per-token V accumulate. JIT mode.
//
// Target: ~5x (previously measured before NaN fix).
// Fix: use scalar max loop instead of reduce<float>(simd, maximum<>()).
// ============================================================================

static constexpr uint32_t HEAD_DIM = 256;
static constexpr uint32_t HD_CHUNKS = HEAD_DIM / 16;
static constexpr uint32_t kBr = 8;
static constexpr uint32_t kBc = 16;

SYCL_ESIMD_FUNCTION inline simd<float, 16> safe_exp16(simd<float, 16> x) {
    simd<float, 16> result = __ESIMD_NS::exp(x);
    result.merge(simd<float, 16>(0.0f), x < -80.0f);
    return result;
}

struct PrefillFMHAArgs {
    fp16* query; fp16* key_cache; fp16* value_cache; fp16* output;
    int32_t* block_table; int32_t* cu_seqlens_q; int32_t* seqused_k;
    float sm_scale;
    uint32_t num_q_heads, num_kv_heads, max_seqlen_q, max_seqlen_k;
    uint32_t block_size, max_blocks_per_seq, batch_size;
    bool is_causal;
};

struct PrefillFMHAKernel_Best {
    PrefillFMHAArgs args;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        uint32_t wg_id = item.get_group(0);
        uint32_t num_q_tiles = (args.max_seqlen_q + kBr - 1) / kBr;
        uint32_t total_per_batch = num_q_tiles * args.num_q_heads;
        uint32_t batch_idx = wg_id / total_per_batch;
        uint32_t remainder = wg_id % total_per_batch;
        uint32_t head_idx = remainder / num_q_tiles;
        uint32_t q_tile_idx = remainder % num_q_tiles;

        if (batch_idx >= args.batch_size) return;
        uint32_t q_start = args.cu_seqlens_q[batch_idx];
        uint32_t seq_len_q = args.cu_seqlens_q[batch_idx + 1] - q_start;
        uint32_t seq_len_k = args.seqused_k[batch_idx];
        uint32_t q_row_start = q_tile_idx * kBr;
        if (q_row_start >= seq_len_q) return;
        uint32_t actual_q_rows = (q_row_start + kBr <= seq_len_q) ? kBr : (seq_len_q - q_row_start);

        uint32_t kv_head_idx = head_idx * args.num_kv_heads / args.num_q_heads;
        uint32_t q_stride = args.num_q_heads * HEAD_DIM;
        uint32_t kv_stride = args.num_kv_heads * HEAD_DIM;

        // Q pointers (load on-the-fly per HD_CHUNK)
        fp16* qp[kBr];
        #pragma unroll
        for (int r = 0; r < kBr; r++)
            qp[r] = args.query + (q_start + q_row_start + r) * q_stride + head_idx * HEAD_DIM;

        // Output (8 rows × 16 HD chunks, hand-unrolled to avoid pointer-array JIT bug)
        simd<float,16> o0[HD_CHUNKS],o1[HD_CHUNKS],o2[HD_CHUNKS],o3[HD_CHUNKS];
        simd<float,16> o4[HD_CHUNKS],o5[HD_CHUNKS],o6[HD_CHUNKS],o7[HD_CHUNKS];
        #pragma unroll
        for (int i=0;i<HD_CHUNKS;i++) {
            o0[i]=0;o1[i]=0;o2[i]=0;o3[i]=0;o4[i]=0;o5[i]=0;o6[i]=0;o7[i]=0;
        }
        float rm0=-1000,rm1=-1000,rm2=-1000,rm3=-1000;
        float rm4=-1000,rm5=-1000,rm6=-1000,rm7=-1000;
        float rs0=0,rs1=0,rs2=0,rs3=0,rs4=0,rs5=0,rs6=0,rs7=0;
        int32_t seq_diff = (int32_t)seq_len_k - (int32_t)seq_len_q;

        // Main loop: kBc=16 tokens per iteration
        for (uint32_t kv_start = 0; kv_start < seq_len_k; kv_start += kBc) {
            if (args.is_causal && (int32_t)kv_start > (int32_t)(q_row_start+actual_q_rows-1)+seq_diff) break;
            uint32_t chunk_len = (kv_start+kBc<=seq_len_k) ? kBc : (seq_len_k-kv_start);

            // ---- DPAS score[8×16] ----
            simd<float,128> score_acc = 0.0f;
            #pragma unroll
            for (int hc=0; hc<HD_CHUNKS; hc++) {
                // Load Q A-tile
                simd<fp16,128> a_tile = 0;
                #pragma unroll
                for (int r=0;r<kBr;r++) {
                    if ((uint32_t)r<actual_q_rows)
                        a_tile.select<16,1>(r*16) = block_load<fp16,16>(qp[r]+hc*16);
                }
                // Load K + VNNI pack
                simd<fp16,256> k_buf = 0;
                #pragma unroll
                for (int c=0;c<kBc;c++) {
                    if ((uint32_t)c<chunk_len) {
                        uint32_t kv_pos=kv_start+c;
                        uint32_t blk=args.block_table[batch_idx*args.max_blocks_per_seq+kv_pos/args.block_size];
                        uint32_t off=kv_pos%args.block_size;
                        k_buf.select<16,1>(c*16)=block_load<fp16,16>(
                            args.key_cache+(uint64_t)blk*args.block_size*kv_stride+off*kv_stride+kv_head_idx*HEAD_DIM+hc*16);
                    }
                }
                auto ku=k_buf.bit_cast_view<uint16_t>();
                simd<uint32_t,128> bv;
                #pragma unroll
                for (int dp=0;dp<8;dp++) {
                    simd<uint16_t,16> lo=ku.select<16,16>(2*dp);
                    simd<uint16_t,16> hi=ku.select<16,16>(2*dp+1);
                    bv.select<16,1>(dp*16)=convert<uint32_t>(lo)|(convert<uint32_t>(hi)<<16);
                }
                score_acc=dpas<8,8,float,float,fp16,fp16>(score_acc,bv.bit_cast_view<fp16>().read(),a_tile);
            }
            score_acc *= args.sm_scale;

            // ---- Extract score rows as simd, mask, SIMD softmax, V accumulate ----
            simd<float,16> sr0=score_acc.select<16,1>(0),  sr1=score_acc.select<16,1>(16);
            simd<float,16> sr2=score_acc.select<16,1>(32), sr3=score_acc.select<16,1>(48);
            simd<float,16> sr4=score_acc.select<16,1>(64), sr5=score_acc.select<16,1>(80);
            simd<float,16> sr6=score_acc.select<16,1>(96), sr7=score_acc.select<16,1>(112);

            // Mask invalid positions
            #pragma unroll
            for (int c=0;c<kBc;c++) {
                if ((uint32_t)c>=chunk_len) { sr0[c]=-1000;sr1[c]=-1000;sr2[c]=-1000;sr3[c]=-1000;sr4[c]=-1000;sr5[c]=-1000;sr6[c]=-1000;sr7[c]=-1000; }
                if (args.is_causal) {
                    int32_t kvp=(int32_t)(kv_start+c);
                    if (kvp>(int32_t)(q_row_start+0)+seq_diff) sr0[c]=-1000;
                    if (kvp>(int32_t)(q_row_start+1)+seq_diff) sr1[c]=-1000;
                    if (kvp>(int32_t)(q_row_start+2)+seq_diff) sr2[c]=-1000;
                    if (kvp>(int32_t)(q_row_start+3)+seq_diff) sr3[c]=-1000;
                    if (kvp>(int32_t)(q_row_start+4)+seq_diff) sr4[c]=-1000;
                    if (kvp>(int32_t)(q_row_start+5)+seq_diff) sr5[c]=-1000;
                    if (kvp>(int32_t)(q_row_start+6)+seq_diff) sr6[c]=-1000;
                    if (kvp>(int32_t)(q_row_start+7)+seq_diff) sr7[c]=-1000;
                }
            }
            if (actual_q_rows<=1) sr1=-1000; if (actual_q_rows<=2) sr2=-1000;
            if (actual_q_rows<=3) sr3=-1000; if (actual_q_rows<=4) sr4=-1000;
            if (actual_q_rows<=5) sr5=-1000; if (actual_q_rows<=6) sr6=-1000;
            if (actual_q_rows<=7) sr7=-1000;

            // Scalar max (workaround for reduce bug)
            float cm0=-1000,cm1=-1000,cm2=-1000,cm3=-1000,cm4=-1000,cm5=-1000,cm6=-1000,cm7=-1000;
            #pragma unroll
            for (int c=0;c<kBc;c++) {
                float v;
                v=sr0.select<1,1>(c)[0]; if(v>cm0)cm0=v;
                v=sr1.select<1,1>(c)[0]; if(v>cm1)cm1=v;
                v=sr2.select<1,1>(c)[0]; if(v>cm2)cm2=v;
                v=sr3.select<1,1>(c)[0]; if(v>cm3)cm3=v;
                v=sr4.select<1,1>(c)[0]; if(v>cm4)cm4=v;
                v=sr5.select<1,1>(c)[0]; if(v>cm5)cm5=v;
                v=sr6.select<1,1>(c)[0]; if(v>cm6)cm6=v;
                v=sr7.select<1,1>(c)[0]; if(v>cm7)cm7=v;
            }

            // SIMD exp (safe) — one instruction per row!
            simd<float,16> p0=safe_exp16(sr0-cm0),p1=safe_exp16(sr1-cm1);
            simd<float,16> p2=safe_exp16(sr2-cm2),p3=safe_exp16(sr3-cm3);
            simd<float,16> p4=safe_exp16(sr4-cm4),p5=safe_exp16(sr5-cm5);
            simd<float,16> p6=safe_exp16(sr6-cm6),p7=safe_exp16(sr7-cm7);

            // Scalar sum (workaround: extract P to array, sum)
            float cs0=0,cs1=0,cs2=0,cs3=0,cs4=0,cs5=0,cs6=0,cs7=0;
            float pa0[kBc],pa1[kBc],pa2[kBc],pa3[kBc],pa4[kBc],pa5[kBc],pa6[kBc],pa7[kBc];
            #pragma unroll
            for (int c=0;c<kBc;c++) {
                pa0[c]=p0.select<1,1>(c)[0]; cs0+=pa0[c];
                pa1[c]=p1.select<1,1>(c)[0]; cs1+=pa1[c];
                pa2[c]=p2.select<1,1>(c)[0]; cs2+=pa2[c];
                pa3[c]=p3.select<1,1>(c)[0]; cs3+=pa3[c];
                pa4[c]=p4.select<1,1>(c)[0]; cs4+=pa4[c];
                pa5[c]=p5.select<1,1>(c)[0]; cs5+=pa5[c];
                pa6[c]=p6.select<1,1>(c)[0]; cs6+=pa6[c];
                pa7[c]=p7.select<1,1>(c)[0]; cs7+=pa7[c];
            }

            // Online correction
            float co0,cn0,co1,cn1,co2,cn2,co3,cn3,co4,cn4,co5,cn5,co6,cn6,co7,cn7;
            #define CORR(RM,CM,CO,CN) { float nm=(RM>CM)?RM:CM; float d1=RM-nm; CO=(d1<-80)?0:sycl::exp(d1); float d2=CM-nm; CN=(d2<-80)?0:sycl::exp(d2); RM=nm; }
            CORR(rm0,cm0,co0,cn0) CORR(rm1,cm1,co1,cn1) CORR(rm2,cm2,co2,cn2) CORR(rm3,cm3,co3,cn3)
            CORR(rm4,cm4,co4,cn4) CORR(rm5,cm5,co5,cn5) CORR(rm6,cm6,co6,cn6) CORR(rm7,cm7,co7,cn7)
            #undef CORR

            rs0=rs0*co0+cs0*cn0; rs1=rs1*co1+cs1*cn1; rs2=rs2*co2+cs2*cn2; rs3=rs3*co3+cs3*cn3;
            rs4=rs4*co4+cs4*cn4; rs5=rs5*co5+cs5*cn5; rs6=rs6*co6+cs6*cn6; rs7=rs7*co7+cs7*cn7;

            // Rescale output
            #pragma unroll
            for (int i=0;i<HD_CHUNKS;i++) {
                o0[i]*=co0;o1[i]*=co1;o2[i]*=co2;o3[i]*=co3;
                o4[i]*=co4;o5[i]*=co5;o6[i]*=co6;o7[i]*=co7;
            }

            // V accumulate (use pre-extracted P values)
            #pragma unroll
            for (int c=0;c<kBc;c++) {
                if ((uint32_t)c>=chunk_len) continue;
                float pv0=pa0[c]*cn0,pv1=pa1[c]*cn1,pv2=pa2[c]*cn2,pv3=pa3[c]*cn3;
                float pv4=pa4[c]*cn4,pv5=pa5[c]*cn5,pv6=pa6[c]*cn6,pv7=pa7[c]*cn7;
                if (pv0==0&&pv1==0&&pv2==0&&pv3==0&&pv4==0&&pv5==0&&pv6==0&&pv7==0) continue;

                uint32_t kv_pos=kv_start+c;
                uint32_t blk=args.block_table[batch_idx*args.max_blocks_per_seq+kv_pos/args.block_size];
                uint32_t off=kv_pos%args.block_size;
                fp16* vp=args.value_cache+(uint64_t)blk*args.block_size*kv_stride+off*kv_stride+kv_head_idx*HEAD_DIM;
                #pragma unroll
                for (int i=0;i<HD_CHUNKS;i++) {
                    simd<float,16> vf=block_load<fp16,16>(vp+i*16);
                    o0[i]+=vf*pv0;o1[i]+=vf*pv1;o2[i]+=vf*pv2;o3[i]+=vf*pv3;
                    o4[i]+=vf*pv4;o5[i]+=vf*pv5;o6[i]+=vf*pv6;o7[i]+=vf*pv7;
                }
            }
        }

        // Write output (hand-unrolled)
        #define WR(R,OUT,RS) if((uint32_t)(R)<actual_q_rows){ \
            float inv=(RS>0)?(1.0f/RS):0.0f; \
            fp16* op=args.output+(q_start+q_row_start+(R))*q_stride+head_idx*HEAD_DIM; \
            _Pragma("unroll") for(int i=0;i<HD_CHUNKS;i++) \
                block_store<fp16,16>(op+i*16,simd<fp16,16>(OUT[i]*inv)); }
        WR(0,o0,rs0) WR(1,o1,rs1) WR(2,o2,rs2) WR(3,o3,rs3)
        WR(4,o4,rs4) WR(5,o5,rs5) WR(6,o6,rs6) WR(7,o7,rs7)
        #undef WR
    }
};

inline void prefill_fmha_launch(
    fp16* query, fp16* key_cache, fp16* value_cache, fp16* output,
    int32_t* block_table, int32_t* cu_seqlens_q, int32_t* seqused_k,
    uint32_t max_seqlen_q, uint32_t max_seqlen_k,
    float sm_scale, bool is_causal,
    uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t block_size, uint32_t max_blocks_per_seq,
    uint32_t batch_size, sycl::queue& q) {
    PrefillFMHAArgs args{query,key_cache,value_cache,output,block_table,cu_seqlens_q,seqused_k,
        sm_scale,num_q_heads,num_kv_heads,max_seqlen_q,max_seqlen_k,block_size,max_blocks_per_seq,batch_size,is_causal};
    uint32_t num_q_tiles=(max_seqlen_q+kBr-1)/kBr;
    uint32_t total_wgs=batch_size*num_q_heads*num_q_tiles;
    q.submit([&](sycl::handler& h){
        PrefillFMHAKernel_Best kernel{args};
        h.parallel_for(sycl::nd_range<1>({total_wgs},{1}),kernel);
    }).wait();
}
