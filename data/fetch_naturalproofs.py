from datasets import load_dataset, Dataset, DatasetDict

def stream_full(path, split):
    it = load_dataset(path, split=split, streaming=True)
    data = list(it)  # iterate all examples
    print(f"{split}: {len(data)} examples")
    return Dataset.from_list(data)

def main():
    train = stream_full("wellecks/naturalproofs-gen", "train")
    valid = stream_full("wellecks/naturalproofs-gen", "validation")
    test  = stream_full("wellecks/naturalproofs-gen", "test")
    dsd = DatasetDict({"train": train, "validation": valid, "test": test})
    dsd.save_to_disk("data/naturalproofs_gen_full")
    print("Saved to data/naturalproofs_gen_full")

if __name__ == "__main__":
    main()
