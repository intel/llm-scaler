# SPDX-License-Identifier: Apache-2.0
import argparse
import gradio as gr
from openai import OpenAI, APIError
from typing import List, Dict, Any, Optional, Tuple
import os
import base64
from pathlib import Path
import cv2
import tempfile
import shutil
from uuid import uuid4

VIDEO_TEMP_DIR = Path("gradio_temp_videos")
if VIDEO_TEMP_DIR.exists():
    shutil.rmtree(VIDEO_TEMP_DIR) 
VIDEO_TEMP_DIR.mkdir()

parser = argparse.ArgumentParser(description='Multimodal Chatbot with Video Support')
parser.add_argument('--model-url', type=str, default='http://localhost:8000/v1', help='Model URL')
parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
parser.add_argument('--temp', type=float, default=0.8, help='Temperature for generation')
parser.add_argument('--stop-token-ids', type=str, default='', help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8003)
args = parser.parse_args()


client = OpenAI(api_key="EMPTY", base_url=args.model_url)


def is_image_file(filename: str) -> bool:
    image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    return any(filename.lower().endswith(ext) for ext in image_exts)

def is_video_file(filename: str) -> bool:
    video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
    return any(filename.lower().endswith(ext) for ext in video_exts)

def encode_file_to_base64(filepath: str) -> str:
    with open(filepath, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')

def extract_frames_from_video(video_path: str, num_frames: int = 10) -> List[str]:
    try:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return []
        
        # <--- MODIFIED: 修正了抽帧逻辑，使用均匀间隔的帧索引
        frame_indices = [int(i) for i in (total_frames / (num_frames + 1) * (j + 1) for j in range(num_frames))]
        temp_files = []
        
        for frame_index in range(total_frames//15):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = video.read()
            if success:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_f:
                    cv2.imwrite(temp_f.name, frame)
                    temp_files.append(temp_f.name)
        video.release()
        return temp_files
    except Exception as e:
        print(f"视频抽帧时发生错误: {e}")
        return []

def predict(messages: List[Dict[str, Any]]):
    """
    调用模型API并以流式返回响应。
    新增了错误处理逻辑。
    """
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temp,
            stream=True,
            extra_body={
                "repetition_penalty": 1.0,
                "stop_token_ids": [int(id) for id in args.stop_token_ids.split(",") if id]
            }
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content, False 
    except APIError as e:
        error_message = f"抱歉，调用模型时出错: {e.message}"
        if "longer than the maximum model length" in e.message:
            error_message = "❌ **输入内容过长** ❌\n\n抱歉，您上传的文本、图片或视频帧的总长度超过了模型的处理上限。请尝试：\n\n- 缩短文字描述\n- 上传尺寸更小的图片\n- 截取更短时间的视频片段"
        
        yield error_message, True 


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Qwen2.5-VL-7B-Instruct Model Serving")
    

    chatbot = gr.Chatbot(height=1200, label="Qwen2.5-VL-7B-Instruct",  avatar_images=("👨", "🤖"), render_markdown=True)
    
    upload_visible = gr.State(False)


    def toggle_upload(visible):
        new_visible = not visible
        return new_visible, gr.Row(visible=new_visible) 
    
    with gr.Group():
        with gr.Row(equal_height=True):
            msg = gr.Textbox(
                placeholder="输入消息...", 
                show_label=False,
                container=False,
                lines=2,
                max_lines=8,
                autofocus=True,
                scale=95
            )
            attach_btn = gr.Button("📎", scale=5) 

        upload_row = gr.Row(visible=False)
        with upload_row:
            file_upload = gr.Files(
                file_types=["image", "video"],
                show_label=False,
                container=False
            )
    attach_btn.click(
        toggle_upload,
        inputs=upload_visible,
        outputs=[upload_visible, upload_row],
        show_progress=False
    )



    with gr.Row():
        submit_btn = gr.Button("🚀 提交", variant="primary")
        clear_btn = gr.Button("🧹 清空")
    
    api_history_state = gr.State([])

    def user_and_bot_response(
        gradio_history: List[Tuple[str, str]],
        api_history: List[Dict[str, Any]],
        user_message: str,
        files: Optional[List[Any]]
    ):
        api_user_content = []
        ui_display_string = ""

        if user_message.strip():
            api_user_content.append({"type": "text", "text": "用中文回答"+user_message.strip()})
            ui_display_string += user_message.strip() + "\n\n"

        if files:
            for file in files:
                filename = file.name
                
                if is_image_file(filename):
                    base64_data = encode_file_to_base64(filename)
                    mime_type = f"image/{Path(filename).suffix[1:].lower()}"
                    data_url = f"data:{mime_type};base64,{base64_data}"
                    ui_display_string += f"![{os.path.basename(filename)}]({data_url})\n"
                    api_user_content.append({"type": "image_url", "image_url": {"url": data_url}})

                elif is_video_file(filename):
                    unique_filename = f"{uuid4()}{Path(filename).suffix}"
                    new_video_path = VIDEO_TEMP_DIR / unique_filename

                    shutil.copyfile(filename, new_video_path)
                    print("Successfully uploaded")

                    with open(new_video_path, "rb") as f:
                        base64_data = base64.b64encode(f.read()).decode()
                    ui_display_string += f"""<video controls width="50%">
                            <source src="data:video/mp4;base64,{base64_data}" type="video/mp4">
                        </video>"""

                    print(ui_display_string)

                    frame_paths = extract_frames_from_video(str(new_video_path), num_frames=10) 
                    if frame_paths:
                        for frame_path in frame_paths:
                            base64_data = encode_file_to_base64(frame_path)
                            api_user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}})
                            os.unlink(frame_path)
                    
        if not api_user_content:
            yield gradio_history, api_history
            return
        api_history.append({"role": "user", "content": api_user_content})
        gradio_history.append((ui_display_string, None))
        yield gradio_history, api_history
        response_stream = predict(api_history)
        full_response = ""
        is_error = False
        for partial_response, error_flag in response_stream:
            full_response += partial_response
            is_error = error_flag
            gradio_history[-1] = (ui_display_string, full_response)
            yield gradio_history, api_history
            if is_error:
                break
        if is_error:
            api_history.pop()
        else:
            api_history.append({"role": "assistant", "content": full_response})
        yield gradio_history, api_history

    def clear_history():
        return [], []

    submit_btn.click(
        user_and_bot_response,
        inputs=[chatbot, api_history_state, msg, file_upload],
        outputs=[chatbot, api_history_state],
        queue=True
    ).then(
       lambda: (gr.Textbox(value=""), gr.Files(value=None)),
       None,
       [msg, file_upload],
       queue=False
    )
    
    clear_btn.click(
        clear_history,
        None,
        [chatbot, api_history_state],
        queue=True
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=True,
        allowed_paths=[str(VIDEO_TEMP_DIR)]
    )