import datasets

_URLS = {
    "train": "data/train-00000-of-00001.jsonl",
    "validation": "data/validation-00000-of-00001.jsonl",
    "test": "data/test-00000-of-00001.jsonl",
}

class RunyoroEnglish(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "runyoro_rutooro": datasets.Value("string"),
                "english": datasets.Value("string"),
            })
        )
    def _split_generators(self, dl_manager):
        downloaded = dl_manager.download(_URLS)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded["validation"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded["test"]}),
        ]
    def _generate_examples(self, filepath):
        import json
        with open(filepath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                yield i, json.loads(line)
