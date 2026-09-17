import json
import random

def convert_to_mixed_cot_sft(input_jsonls, output_jsonl, prompt_text):
    converted_data = []
    
    for input_jsonl in input_jsonls:
        with open(input_jsonl, 'r', encoding='utf-8') as fin:
            for line in fin:
                item = json.loads(line)
                audio_path = item["audio"]["path"]
                sentences = item["sentences"]
                
                spk_count = len(sentences) # 动态识别是 2 人还是 3 人
                
                refs = []
                for s in sentences:
                    formatted_s = f"[{float(s['start']):.1f}s - {float(s['end']):.1f}s] {s['text'].lower()}"
                    refs.append(formatted_s)
                
                # 随机打乱真实顺序，强迫模型靠听觉而非死记硬背来解耦
                random.shuffle(refs)

                # 💡 组装带有 CoT 的终极目标文本
                cot_text = f"<think> estimated speaker-number: {spk_count} </think> "
                spk_text = " ".join([f"<spk{i+1}> {refs[i]} </spk{i+1}>" for i in range(spk_count)])
                
                target_text = cot_text + spk_text

                swift_item = {
                    "query": f"<audio>{prompt_text}",
                    "response": target_text,
                    "audios": [audio_path]
                }
                converted_data.append(swift_item)
            
    # 全局打乱，让 2人和3人的数据均匀交替
    random.shuffle(converted_data)
            
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for item in converted_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ 成功混合并转换 {len(converted_data)} 条 CoT SFT 数据至 {output_jsonl}")

if __name__ == "__main__":
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
        "./dataset/libri2mix_train.jsonl",
        "./dataset/libri3mix_train.jsonl"
    ]
    OUTPUT_PATH = "mixed_cot_sft_train.jsonl"
    
    convert_to_mixed_cot_sft(INPUT_PATHS, OUTPUT_PATH, PROMPT)
