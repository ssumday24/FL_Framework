"""Exercise the actual backend's generation path with fake tensors/tokenizer."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from fl_framework.config import ModelConfig
from fl_framework.models import TextModel


class Batch(dict):
    def to(self, device):
        self.device = device
        return self


class GenerationTests(unittest.TestCase):
    def model(self, prompt_format='chat', context=100):
        model = TextModel.__new__(TextModel)
        model.config = ModelConfig('fake', prompt_format=prompt_format,
                                   max_input_tokens=50, max_new_tokens=10)
        model.torch = SimpleNamespace(inference_mode=nullcontext)
        model.tokenizer = MagicMock()
        model.tokenizer.pad_token_id = 2
        model.tokenizer.apply_chat_template.return_value = 'CHAT WITH BOS'
        tensor = MagicMock()
        tensor.shape = (1, 20)
        batch = Batch(input_ids=tensor, attention_mask=object())
        model.tokenizer.return_value = batch
        model.tokenizer.decode.return_value = ' answer only '
        model.model = MagicMock()
        model.model.config.max_position_embeddings = context
        model.model.get_input_embeddings.return_value.weight.device = 'cuda:0'
        return model, batch

    def test_chat_does_not_duplicate_bos_and_decodes_only_new_tokens(self):
        model, batch = self.model()
        result = model.generate('question')
        self.assertEqual(result, 'answer only')
        model.tokenizer.assert_called_once_with('CHAT WITH BOS', add_special_tokens=False,
                                                return_tensors='pt', truncation=False)
        generated = model.model.generate.return_value
        generated.__getitem__.assert_called_once_with((0, slice(20, None)))
        self.assertEqual(batch.device, 'cuda:0')
        self.assertFalse(model.model.generate.call_args.kwargs['do_sample'])
        self.assertNotIn('temperature', model.model.generate.call_args.kwargs)
        self.assertIn('attention_mask', model.model.generate.call_args.kwargs)

    def test_prometheus_receives_mistral_instruction_wrapper(self):
        model, _ = self.model('prometheus')
        model.generate('criterion and candidate')
        prompt = model.tokenizer.call_args.args[0]
        self.assertTrue(prompt.startswith('[INST] '))
        self.assertTrue(prompt.endswith(' [/INST]'))
        self.assertIn('criterion and candidate', prompt)
        self.assertTrue(model.tokenizer.call_args.kwargs['add_special_tokens'])

    def test_overlong_input_stops_before_generation(self):
        model, _ = self.model(context=25)
        with self.assertRaisesRegex(ValueError, 'no silent truncation'):
            model.generate('question')
        model.model.generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
