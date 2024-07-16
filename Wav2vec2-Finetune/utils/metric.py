from evaluate import load
import torch

class Metric:
    def __init__(self, processor):
        self.processor = processor
        self.wer_metric = load("wer")
        self.cer_metric = load("cer")

    def __call__(self, logits, labels):
        preds = torch.argmax(logits, axis=-1)
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        pred_strs = self.processor.batch_decode(preds)        
        # we do not want to group tokens when computing the metrics
        label_strs = self.processor.batch_decode(labels, group_tokens=False)
        
        wer = self.wer_metric.compute(predictions=pred_strs, references=label_strs)
        cer = self.cer_metric.compute(predictions=pred_strs, references=label_strs)
        
        return {'wer': wer, 'cer': cer}
