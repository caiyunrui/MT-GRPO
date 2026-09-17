import json

def prepare_mixed_cot_grpo(input_jsonls, output_jsonl, prompt_text):
    converted_data = []
    
    for input_jsonl in input_jsonls:
        with open(input_jsonl, 'r', encoding='utf-8') as fin:
            for line in fin:
                item = json.loads(line)
                audio_path = item["audio"]["path"]
                sentences = item["sentences"]
                spk_count = len(sentences)
                
                # 统一预留 3 个坑位，如果只有 2 人，ref_c 就会保持空字符串 ""
                refs = ["", "", ""]
                for i, s in enumerate(sentences):
                    refs[i] = f"[{float(s['start']):.1f}s - {float(s['end']):.1f}s] {s['text'].lower()}"

                swift_item = {
                    "query": f"<audio>{prompt_text}",
                    "audios": [audio_path],
                    "ref_a": refs[0],
                    "ref_b": refs[1],
                    "ref_c": refs[2], # 2人数据此处自动为空
                    "spk_count": spk_count # 💡 透传给 Reward 引擎用来给 CoT 打分
                }
                converted_data.append(swift_item)
                
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for item in converted_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ 成功混合提取 {len(converted_data)} 条带 CoT 提示的 GRPO 数据至 {output_jsonl}")

if __name__ == "__main__":
    # Prompt 必须和 SFT 保持完全一致！
    PROMPT = (
        "Transcribe the overlapping audio and provide precise speaker timestamps. "
        "First, estimate the number of speakers using EXACTLY this format: <think> estimated speaker-number: N </think>. "
        "Then, format your transcription EXACTLY as: "
        "<spk1> [start_time - end_time] words </spk1> ... up to the actual number of speakers. "
        "Crucial Rules: "
        "1. Timestamps must be in seconds to 1 decimal place (e.g., [0.5s - 2.1s]). "
        "2. Do not hallucinate extra speakers or words."
    )

    INPUT_PATHS = [
        "/mmu-audio-ssd/yaron/SpeechLLM/dataset/libri2mix_test.jsonl",
        "/mmu-audio-ssd/yaron/SpeechLLM/dataset/libri3mix_test.jsonl"
    ]
    OUTPUT_PATH = "swift_mixed_cot_grpo_test.jsonl"
    
    prepare_mixed_cot_grpo(INPUT_PATHS, OUTPUT_PATH, PROMPT)
