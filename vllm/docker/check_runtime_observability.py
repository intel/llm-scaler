#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export a real OTLP span to a local receiver; requires no GPU or network."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCSpanExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class Receiver(BaseHTTPRequestHandler):
    spans = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        request = ExportTraceServiceRequest.FromString(body)
        self.spans.extend(span.name for resource in request.resource_spans
                          for scope in resource.scope_spans for span in scope.spans)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    # Exercise the other observability imports as well as HTTP export.
    grpc = GRPCSpanExporter(endpoint="127.0.0.1:1", insecure=True)
    grpc.shutdown()
    prometheus = PrometheusMetricReader()
    prometheus.shutdown()
    server = HTTPServer(("127.0.0.1", 0), Receiver)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces",
                                timeout=5)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with provider.get_tracer("runtime-build-check").start_as_current_span("xpu-runtime-smoke"):
            pass
        assert provider.force_flush(timeout_millis=5000)
        assert Receiver.spans == ["xpu-runtime-smoke"], Receiver.spans
    finally:
        provider.shutdown()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    print("OTLP HTTP export and gRPC/Prometheus imports passed")


if __name__ == "__main__":
    main()
