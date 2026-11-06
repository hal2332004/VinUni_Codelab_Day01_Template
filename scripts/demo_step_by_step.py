"""
Demo Step-by-Step for Lab 01 Functions.
Runs each task step-by-step, showing input arguments and returned output.
Can run in MOCK mode (no API key needed) or LIVE mode (if API keys are in .env).
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Thêm thư mục gốc vào path để import
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import code từ starter-code/template.py hoặc solution-code/solution.py
from solution import (
    call_openai,
    call_gemini,
    call_anthropic,
    compare_models,
    retry_with_backoff,
    batch_compare,
    format_comparison_table,
    PRICING_1M_TOKENS,
)


def press_to_continue(step_name: str):
    print("\n" + "=" * 60)
    print(f"👉 [BƯỚC TIẾP THEO] {step_name}")
    print("=" * 60)
    input("Nhấn ENTER để chạy bước này...")


def main():
    print("\n" + "#" * 60)
    print("#  CHƯƠNG TRÌNH CHẠY THỬ TỪNG TESTCASE & KẾT QUẢ TỪNG BƯỚC  #")
    print("#" * 60)

    # ---------------------------------------------------------
    # BƯỚC 1: Task 1 - call_openai
    # ---------------------------------------------------------
    press_to_continue("Task 1: call_openai")
    prompt1 = "Xin chào, hãy giới thiệu ngắn về AI trong 1 câu."
    print(f"Prompt gửi đi: '{prompt1}'")

    with patch("openai.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="AI là trí tuệ do con người lập trình cho máy móc mô phỏng."))]
        mock_resp.usage = MagicMock(prompt_tokens=15, completion_tokens=22)
        mock_client.chat.completions.create.return_value = mock_resp

        text, latency, usage = call_openai(prompt1)
        print("\nKết quả trả về từ hàm call_openai:")
        print(f"  - text: {text}")
        print(f"  - latency: {latency:.4f}s")
        print(f"  - usage: {usage}")

    # ---------------------------------------------------------
    # BƯỚC 2: Task 2 - call_gemini
    # ---------------------------------------------------------
    press_to_continue("Task 2: call_gemini")
    prompt2 = "Tại sao Gemini 2.5 Flash lại nhanh?"
    print(f"Prompt gửi đi: '{prompt2}'")

    with patch("google.genai.Client") as MockGenAI:
        mock_client = MagicMock()
        MockGenAI.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.text = "Gemini 2.5 Flash được tối ưu hóa kiến trúc mạng để giảm độ trễ và chi phí suy luận."
        mock_resp.usage_metadata = MagicMock(prompt_token_count=12, candidates_token_count=28)
        mock_client.models.generate_content.return_value = mock_resp

        text, latency, usage = call_gemini(prompt2)
        print("\nKết quả trả về từ hàm call_gemini:")
        print(f"  - text: {text}")
        print(f"  - latency: {latency:.4f}s")
        print(f"  - usage: {usage}")

    # ---------------------------------------------------------
    # BƯỚC 3: Task 3 - call_anthropic
    # ---------------------------------------------------------
    press_to_continue("Task 3: call_anthropic")
    prompt3 = "Claude 3.5 Haiku có ưu điểm gì?"
    print(f"Prompt gửi đi: '{prompt3}'")

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Claude 3.5 Haiku có tốc độ phản hồi cực nhanh với chi phí rất tiết kiệm.")]
        mock_resp.usage = MagicMock(input_tokens=14, output_tokens=26)
        mock_client.messages.create.return_value = mock_resp

        text, latency, usage = call_anthropic(prompt3)
        print("\nKết quả trả về từ hàm call_anthropic:")
        print(f"  - text: {text}")
        print(f"  - latency: {latency:.4f}s")
        print(f"  - usage: {usage}")

    # ---------------------------------------------------------
    # BƯỚC 4: Task 4 - compare_models
    # ---------------------------------------------------------
    press_to_continue("Task 4: compare_models (So sánh GPT-4o, GPT-4o-Mini, Gemini Flash)")
    prompt4 = "So sánh sự khác biệt giữa Supervised và Unsupervised Learning."
    print(f"Prompt gửi đi: '{prompt4}'")

    with patch("starter-code.template.call_openai") if "starter-code" in sys.modules else patch("solution.call_openai") as mock_oa, \
         patch("starter-code.template.call_gemini") if "starter-code" in sys.modules else patch("solution.call_gemini") as mock_gem:

        mock_oa.side_effect = [
            ("Phản hồi chi tiết từ GPT-4o...", 0.65, {"input_tokens": 18, "output_tokens": 45}),
            ("Phản hồi nhanh gọn từ GPT-4o-Mini...", 0.32, {"input_tokens": 18, "output_tokens": 40}),
        ]
        mock_gem.return_value = ("Phản hồi siêu tốc từ Gemini 2.5 Flash...", 0.25, {"input_tokens": 18, "output_tokens": 42})

        comp_result = compare_models(prompt4)
        print("\nKết quả trả về dạng Dictionary có cấu trúc chi phí (cost) USD:")
        for model_key, data in comp_result.items():
            print(f"\n  [{model_key}]")
            print(f"    - Response: {data['response']}")
            print(f"    - Latency: {data['latency']}s")
            print(f"    - Tokens (In/Out): {data['input_tokens']} / {data['output_tokens']}")
            print(f"    - Cost (USD): ${data['cost']:.8f}")

    # ---------------------------------------------------------
    # BƯỚC 5: Bonus Task A - retry_with_backoff
    # ---------------------------------------------------------
    press_to_continue("Bonus Task A: retry_with_backoff (Tự động thử lại khi gặp lỗi)")
    print("Mô phỏng: Một hàm gọi API bị lỗi 2 lần đầu (Rate Limit), lần thứ 3 mới thành công.")

    call_count = 0
    def unstable_api():
        nonlocal call_count
        call_count += 1
        print(f"    -> Đang thực thi lần thử #{call_count}...")
        if call_count < 3:
            raise ConnectionError("Lỗi 429: Rate Limit Exceeded!")
        return "Kết nối thành công sau 2 lần retry!"

    result = retry_with_backoff(unstable_api, max_retries=3, base_delay=0.1)
    print(f"\nKết quả nhận được: '{result}'")
    print(f"Tổng số lần đã thử: {call_count}")

    # ---------------------------------------------------------
    # BƯỚC 6: Bonus Task B & C - batch_compare & format_comparison_table
    # ---------------------------------------------------------
    press_to_continue("Bonus Task B & C: batch_compare và format_comparison_table")
    sample_prompts = [
        "Python là gì?",
        "Machine Learning là gì?"
    ]
    print(f"Danh sách prompt chạy batch: {sample_prompts}")

    with patch("solution.compare_models") if "solution" in sys.modules else patch("starter-code.template.compare_models") as mock_cm:
        def fake_compare(p):
            return {
                "gpt4o": {"response": f"GPT-4o trả lời cho: {p}", "latency": 0.55, "cost": 0.00045, "input_tokens": 10, "output_tokens": 20},
                "gpt4o_mini": {"response": f"Mini trả lời cho: {p}", "latency": 0.28, "cost": 0.0000135, "input_tokens": 10, "output_tokens": 20},
                "gemini_flash": {"response": f"Gemini trả lời cho: {p}", "latency": 0.22, "cost": 0.00000675, "input_tokens": 10, "output_tokens": 20},
            }
        mock_cm.side_effect = fake_compare

        batch_res = batch_compare(sample_prompts)
        table_output = format_comparison_table(batch_res)

        print("\nBảng Markdown so sánh kết quả tự động:")
        print(table_output)

    print("\n" + "=" * 60)
    print("🎉 ĐÃ HOÀN THÀNH TẤT CẢ CÁC BƯỚC THỬ NGHIỆM THÀNH CÔNG!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

