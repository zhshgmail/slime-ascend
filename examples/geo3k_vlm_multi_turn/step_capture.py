"""Opt-in original environment objects in the existing native Sample trace.

No scorer, inference, input replay, resource acquisition or acceptance decisions.
"""
from copy import deepcopy
import hashlib
from pathlib import Path


class StepCapture:
    def __init__(self, sample, identity, source_paths):
        if not isinstance(identity, dict) or any(
            not isinstance(identity.get(k), str) or not identity[k] for k in ('input', 'cohort')
        ):
            raise ValueError('capture identity requires explicit input and cohort references')
        if 'env_step_capture' in sample.metadata:
            raise ValueError('refusing to replace an existing environment capture')
        self.sample = sample
        self.record = {
            'schema': 'original-env-step.v1', 'identity': deepcopy(identity),
            'sources': {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in source_paths},
            'sample': {'index': sample.index, 'group_index': sample.group_index, 'group_id': sample.group_id},
            'steps': [],
        }
        sample.metadata['env_step_capture'] = self.record
        self.current = None

    def begin(self, turn, response_text, response_token_count):
        if self.current is not None and not self.current['received']:
            raise ValueError('previous original environment return was not received')
        if self.record['steps'] and turn <= self.record['steps'][-1]['turn']:
            raise ValueError('environment turn order changed')
        self.current = {'turn': turn, 'response_text': response_text,
                        'response_token_range': [len(self.sample.tokens) - response_token_count, len(self.sample.tokens)],
                        'response_mask_range': [len(self.sample.loss_mask) - response_token_count, len(self.sample.loss_mask)],
                        'produced': None, 'received': False,
                        'observation_token_range': None, 'observation_mask_range': None}
        self.record['steps'].append(self.current)

    def produced(self, response_text, observation, done, info, tool_record):
        if self.current is None or self.current['produced'] is not None:
            raise ValueError('missing or duplicate original producer boundary')
        if response_text != self.current['response_text']:
            raise ValueError('producer response does not match this turn')
        self._produced_return = (observation, done, info)
        self.current['produced'] = deepcopy({'observation': observation, 'done': done,
                                            'info': info, 'tool_record': tool_record})

    def received(self, observation, done, info):
        if self.current is None or self.current['produced'] is None or self.current['received']:
            raise ValueError('missing or duplicate environment return boundary')
        if (observation is not self._produced_return[0] or done != self._produced_return[1]
                or info is not self._produced_return[2]):
            raise ValueError('dispatcher dropped or replaced original environment return objects')
        self._produced_return = None
        self.current['return'] = deepcopy({'observation': observation, 'done': done, 'info': info})
        self.current['received'] = True

    def encoded(self, message, tokens):
        self.current['formatted_observation'] = deepcopy(message)
        self.current['encoded_observation_tokens'] = deepcopy(tokens)

    def appended(self, token_count):
        self.current['observation_token_range'] = [len(self.sample.tokens) - token_count, len(self.sample.tokens)]
        self.current['observation_mask_range'] = [len(self.sample.loss_mask) - token_count, len(self.sample.loss_mask)]
