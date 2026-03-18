"""Gradio Web UI for Diogenes Model Testing.

A web-based interface for interactive model testing, comparison, and result visualization.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from diogenes.model import DiogenesModel, EpistemicMode
from diogenes.inference import DiogenesInference, InferenceResult

from diogenes.testing.core.storage import TestStorage, TestResult
from diogenes.testing.core.runner import TestRunner, TestConfig, TestSuite


logger = logging.getLogger(__name__)


# Global state
class AppState:
    """Application state management."""

    def __init__(self):
        self.model: Optional[DiogenesModel] = None
        self.model_name: str = ""
        self.inference: Optional[DiogenesInference] = None
        self.storage: Optional[TestStorage] = None
        self.history: list[dict[str, Any]] = []

    def load_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        use_4bit: bool = False,
    ) -> str:
        """Load the model."""
        try:
            self.model = DiogenesModel.from_pretrained(
                model_name_or_path=model_path or model_name,
                use_4bit=use_4bit,
            )
            self.inference = DiogenesInference(self.model)
            self.model_name = model_name
            return f"✅ Model loaded: {model_name}"
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return f"❌ Failed to load model: {str(e)}"

    def set_storage(self, storage_path: str, backend: str = "jsonl") -> str:
        """Set up storage."""
        try:
            if self.storage:
                self.storage.close()
            self.storage = TestStorage(storage_path, backend)
            return f"✅ Storage initialized: {storage_path}"
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            return f"❌ Failed to initialize storage: {str(e)}"

    def add_to_history(self, result: dict[str, Any]) -> None:
        """Add result to history."""
        self.history.append(result)
        # Keep only last 100 results
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def get_history_dataframe(self) -> list[list[Any]]:
        """Get history as list of lists for DataFrame."""
        rows = []
        for item in self.history:
            rows.append([
                item.get("timestamp", "")[:19],
                item.get("prompt", "")[:100],
                item.get("mode", ""),
                item.get("confidence", 0.0),
                item.get("latency_ms", 0.0),
            ])
        return rows


# Create global state
state = AppState()


def run_inference(
    prompt: str,
    temperature: float,
    max_length: int,
    top_p: float,
) -> tuple[str, str, str, str]:
    """Run inference on a prompt.

    Returns:
        Tuple of (response, mode, confidence, latency)
    """
    if state.inference is None:
        return (
            "⚠️ Please load a model first using the Model tab.",
            "",
            "",
            "",
        )

    import time

    start = time.time()

    try:
        result: InferenceResult = state.inference.generate(
            prompt=prompt,
            temperature=temperature,
            max_length=max_length,
            top_p=top_p,
            return_logprobs=True,
        )

        latency_ms = (time.time() - start) * 1000

        # Create result record
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "response": result.text,
            "mode": result.epistemic_mode.value,
            "confidence": result.confidence,
            "latency_ms": latency_ms,
        }

        # Save to storage if available
        if state.storage:
            test_result = TestResult(
                test_id=datetime.utcnow().isoformat(),
                prompt=prompt,
                response=result.text,
                epistemic_mode=result.epistemic_mode.value,
                confidence=result.confidence,
                latency_ms=latency_ms,
                token_count=len(result.tokens),
            )
            state.storage.save(test_result)

        # Add to history
        state.add_to_history(record)

        # Format response
        mode_emoji = {
            EpistemicMode.DIRECT_ANSWER.value: "✅",
            EpistemicMode.CAUTIOUS_LIMIT.value: "⚠️",
            EpistemicMode.ABSTAIN.value: "🚫",
            EpistemicMode.CLARIFY.value: "❓",
            EpistemicMode.REJECT_PREMISE.value: "❌",
            EpistemicMode.REQUEST_TOOL.value: "🔧",
            EpistemicMode.PROBABILISTIC.value: "🎲",
        }

        mode_display = f"{mode_emoji.get(result.epistemic_mode.value, '')} {result.epistemic_mode.value}"

        return (
            result.text,
            mode_display,
            f"{result.confidence:.4f}",
            f"{latency_ms:.2f} ms",
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return (
            f"❌ Error: {str(e)}",
            "",
            "",
            "",
        )


def run_comparison(
    prompt: str,
    model_a_path: str,
    model_b_path: str,
    temperature: float,
    max_length: int,
) -> tuple[str, str, str, str, str, str]:
    """Run comparison between two models.

    Returns:
        Tuple of (response_a, mode_a, conf_a, response_b, mode_b, conf_b)
    """
    try:
        # Load models temporarily
        model_a = DiogenesModel.from_pretrained(model_a_path, use_4bit=False)
        model_b = DiogenesModel.from_pretrained(model_b_path, use_4bit=False)

        inf_a = DiogenesInference(model_a)
        inf_b = DiogenesInference(model_b)

        # Run inference
        result_a: InferenceResult = inf_a.generate(
            prompt=prompt,
            temperature=temperature,
            max_length=max_length,
        )
        result_b: InferenceResult = inf_b.generate(
            prompt=prompt,
            temperature=temperature,
            max_length=max_length,
        )

        return (
            result_a.text,
            f"{result_a.epistemic_mode.value} ({result_a.confidence:.4f})",
            f"{result_a.confidence:.4f}",
            result_b.text,
            f"{result_b.epistemic_mode.value} ({result_b.confidence:.4f})",
            f"{result_b.confidence:.4f}",
        )

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        error_msg = f"❌ Error: {str(e)}"
        return (error_msg, "", "", error_msg, "", "")


def refresh_history() -> list[list[Any]]:
    """Refresh the history table."""
    return state.get_history_dataframe()


def load_suite(suite_path: str) -> str:
    """Load a test suite."""
    try:
        suite = TestSuite.from_json(suite_path)
        return f"✅ Loaded suite: {suite.name} ({len(suite.test_cases)} test cases)"
    except Exception as e:
        logger.error(f"Failed to load suite: {e}")
        return f"❌ Failed to load suite: {str(e)}"


def run_suite(
    suite_path: str,
    temperature: float,
    max_length: int,
    parallel: bool,
) -> str:
    """Run a complete test suite."""
    if state.inference is None:
        return "⚠️ Please load a model first."

    try:
        suite = TestSuite.from_json(suite_path)

        config = TestConfig(
            model_name=state.model_name,
            temperature=temperature,
            max_length=max_length,
        )

        runner = TestRunner(model=state.model, config=config)

        results = runner.run_suite(suite=suite, parallel=parallel)

        runner.close()

        return f"✅ Completed {len(results)} tests from suite '{suite.name}'"

    except Exception as e:
        logger.error(f"Suite execution error: {e}")
        return f"❌ Error: {str(e)}"


def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(title="Diogenes Model Test Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🧪 Diogenes Model Test Tool

            Comprehensive testing interface for evaluating Diogenes model performance
            and epistemic mode detection.
            """
        )

        with gr.Tabs():
            # =====================================================================
            # Model Tab
            # =====================================================================
            with gr.TabItem("🤖 Model"):
                gr.Markdown("### Load and Configure Model")

                with gr.Row():
                    with gr.Column():
                        model_name_input = gr.Textbox(
                            label="Model Name or Path",
                            placeholder="Qwen/Qwen3-0.6B or ./models/my-model",
                            value="Qwen/Qwen3-0.6B",
                        )
                        use_4bit = gr.Checkbox(
                            label="Use 4-bit Quantization",
                            value=False,
                        )
                        load_btn = gr.Button("Load Model", variant="primary")

                    with gr.Column():
                        model_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column():
                        storage_path_input = gr.Textbox(
                            label="Storage Path",
                            placeholder="./test_results",
                            value="./test_results",
                        )
                        storage_backend = gr.Radio(
                            choices=["jsonl", "sqlite"],
                            value="jsonl",
                            label="Storage Backend",
                        )
                        save_storage_btn = gr.Button("Initialize Storage")

                    with gr.Column():
                        storage_status = gr.Textbox(
                            label="Storage Status",
                            interactive=False,
                        )

                load_btn.click(
                    fn=state.load_model,
                    inputs=[model_name_input, gr.Textbox(value=None), use_4bit],
                    outputs=[model_status],
                )

                save_storage_btn.click(
                    fn=state.set_storage,
                    inputs=[storage_path_input, storage_backend],
                    outputs=[storage_status],
                )

            # =====================================================================
            # Quick Test Tab
            # =====================================================================
            with gr.TabItem("⚡ Quick Test"):
                gr.Markdown("### Single Prompt Testing")

                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=4,
                        )

                    with gr.Column(scale=1):
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                        )
                        max_length_slider = gr.Slider(
                            minimum=64,
                            maximum=2048,
                            value=512,
                            step=64,
                            label="Max Length",
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top P",
                        )
                        run_btn = gr.Button("Run Test", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        response_output = gr.Textbox(
                            label="Response",
                            lines=6,
                        )

                    with gr.Column():
                        mode_output = gr.Textbox(label="Epistemic Mode")
                        confidence_output = gr.Textbox(label="Confidence")
                        latency_output = gr.Textbox(label="Latency")

                run_btn.click(
                    fn=run_inference,
                    inputs=[prompt_input, temperature_slider, max_length_slider, top_p_slider],
                    outputs=[response_output, mode_output, confidence_output, latency_output],
                )

            # =====================================================================
            # Comparison Tab
            # =====================================================================
            with gr.TabItem("🔄 Comparison"):
                gr.Markdown("### Side-by-Side Model Comparison")

                with gr.Row():
                    with gr.Column():
                        model_a_input = gr.Textbox(
                            label="Model A Path",
                            placeholder="./models/model_a",
                        )
                        model_b_input = gr.Textbox(
                            label="Model B Path",
                            placeholder="./models/model_b",
                        )
                        compare_temp_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                        )
                        compare_max_len_slider = gr.Slider(
                            minimum=64,
                            maximum=2048,
                            value=512,
                            step=64,
                            label="Max Length",
                        )

                compare_prompt_input = gr.Textbox(
                    label="Prompt for Comparison",
                    placeholder="Enter prompt to compare both models...",
                    lines=3,
                )
                compare_btn = gr.Button("Compare Models", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Model A Response")
                        response_a_output = gr.Textbox(label="Response A", lines=6)
                        mode_a_output = gr.Textbox(label="Mode A")
                        conf_a_output = gr.Textbox(label="Confidence A")

                    with gr.Column():
                        gr.Markdown("#### Model B Response")
                        response_b_output = gr.Textbox(label="Response B", lines=6)
                        mode_b_output = gr.Textbox(label="Mode B")
                        conf_b_output = gr.Textbox(label="Confidence B")

                compare_btn.click(
                    fn=run_comparison,
                    inputs=[
                        compare_prompt_input,
                        model_a_input,
                        model_b_input,
                        compare_temp_slider,
                        compare_max_len_slider,
                    ],
                    outputs=[
                        response_a_output,
                        mode_a_output,
                        conf_a_output,
                        response_b_output,
                        mode_b_output,
                        conf_b_output,
                    ],
                )

            # =====================================================================
            # Batch Test Tab
            # =====================================================================
            with gr.TabItem("📦 Batch Test"):
                gr.Markdown("### Batch Testing with Test Suites")

                with gr.Row():
                    with gr.Column():
                        suite_path_input = gr.Textbox(
                            label="Test Suite Path",
                            placeholder="./suites/epistemic_modes.json",
                        )
                        suite_temp_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                        )
                        suite_max_len_slider = gr.Slider(
                            minimum=64,
                            maximum=2048,
                            value=512,
                            step=64,
                            label="Max Length",
                        )
                        suite_parallel = gr.Checkbox(
                            label="Parallel Execution",
                            value=True,
                        )

                    with gr.Column():
                        load_suite_btn = gr.Button("Load Suite")
                        run_suite_btn = gr.Button("Run Suite", variant="primary")
                        suite_status_output = gr.Textbox(label="Status", interactive=False)

                load_suite_btn.click(
                    fn=load_suite,
                    inputs=[suite_path_input],
                    outputs=[suite_status_output],
                )

                run_suite_btn.click(
                    fn=run_suite,
                    inputs=[
                        suite_path_input,
                        suite_temp_slider,
                        suite_max_len_slider,
                        suite_parallel,
                    ],
                    outputs=[suite_status_output],
                )

            # =====================================================================
            # History Tab
            # =====================================================================
            with gr.TabItem("📜 History"):
                gr.Markdown("### Test History")

                refresh_btn = gr.Button("Refresh History")

                history_table = gr.Dataframe(
                    headers=["Timestamp", "Prompt", "Mode", "Confidence", "Latency"],
                    label="Test History",
                    interactive=False,
                )

                refresh_btn.click(
                    fn=refresh_history,
                    outputs=[history_table],
                )

                # Auto-refresh on tab click
                app.load(
                    fn=refresh_history,
                    outputs=[history_table],
                )

        # Footer
        gr.Markdown(
            """
            ---
            **Diogenes Model Test Tool** v0.1.0 | Built with Gradio
            """
        )

    return app


def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    **kwargs,
) -> None:
    """Launch the Gradio application.

    Args:
        server_name: Server hostname
        server_port: Server port
        share: Create public share link
        **kwargs: Additional arguments for launch()
    """
    app = create_app()
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        **kwargs,
    )


if __name__ == "__main__":
    launch_app()
