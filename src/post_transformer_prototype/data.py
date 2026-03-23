"""Synthetic long-context tasks for the post-transformer prototype."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset

PAD = 0
PASS = 1
MAP = 2
STEP = 3
ASK = 4
SEP = 5

KEY_START = 8
NUM_KEYS = 12
ALIAS_START = KEY_START + NUM_KEYS
VALUE_START = ALIAS_START + NUM_KEYS
BRIDGE_START = VALUE_START + NUM_KEYS
NOISE_START = BRIDGE_START + NUM_KEYS
VOCAB_SIZE = 256
NUM_VALUES = NUM_KEYS

TEXT_PAD = 0
TEXT_BOS = 1
TEXT_EOS = 2
TEXT_PERIOD = 3
TEXT_QMARK = 4
TEXT_ENTITY = 5
TEXT_ALIAS = 6
TEXT_BRIDGE = 7
TEXT_VALUE = 8
TEXT_MAPS = 9
TEXT_TO = 10
TEXT_LINKS = 11
TEXT_REFERS = 12
TEXT_QUESTION = 13
TEXT_WHAT = 14
TEXT_IS = 15
TEXT_OF = 16
TEXT_ANSWER = 17
TEXT_AND = 18
TEXT_FOCUS = 19
TEXT_NAME_KEY_START = 32
TEXT_NAME_ALIAS_START = TEXT_NAME_KEY_START + NUM_KEYS
TEXT_NAME_BRIDGE_START = TEXT_NAME_ALIAS_START + NUM_KEYS
TEXT_NAME_VALUE_START = TEXT_NAME_BRIDGE_START + NUM_KEYS
TEXT_NOISE_START = TEXT_NAME_VALUE_START + NUM_VALUES
TEXT_NOISE_WORDS = 24
TEXT_VOCAB_SIZE = TEXT_NOISE_START + TEXT_NOISE_WORDS


@dataclass(frozen=True)
class Sample:
    task: str
    context_tokens: List[int]
    query_tokens: List[int]
    answer_id: int


@dataclass(frozen=True)
class TextGraphSample:
    task: str
    sequence_tokens: List[int]
    answer_token: int
    answer_position: int
    map_target: torch.Tensor
    step_target: torch.Tensor


def answer_token(answer_id: int) -> int:
    return VALUE_START + answer_id


def alias_for_key(key_id: int) -> int:
    return ALIAS_START + key_id


def key_token(key_id: int) -> int:
    return KEY_START + key_id


def bridge_token(bridge_id: int) -> int:
    return BRIDGE_START + bridge_id


def text_key_name(key_id: int) -> int:
    return TEXT_NAME_KEY_START + key_id


def text_alias_name(key_id: int) -> int:
    return TEXT_NAME_ALIAS_START + key_id


def text_bridge_name(bridge_id: int) -> int:
    return TEXT_NAME_BRIDGE_START + bridge_id


def text_value_name(value_id: int) -> int:
    return TEXT_NAME_VALUE_START + value_id


def text_noise_word(rng: random.Random) -> int:
    return rng.randint(TEXT_NOISE_START, TEXT_NOISE_START + TEXT_NOISE_WORDS - 1)


def random_noise(rng: random.Random) -> int:
    return rng.randint(NOISE_START, VOCAB_SIZE - 1)


def choose_insert_positions(
    rng: random.Random,
    total_length: int,
    segment_lengths: Sequence[int],
) -> List[int]:
    positions: List[int] = []
    occupied = set()
    for seg_len in segment_lengths:
        candidates = [
            pos
            for pos in range(total_length - seg_len + 1)
            if all((pos + offset) not in occupied for offset in range(seg_len))
        ]
        if not candidates:
            raise ValueError("Context length too small for requested segments.")
        pos = rng.choice(candidates)
        positions.append(pos)
        for offset in range(seg_len):
            occupied.add(pos + offset)
    return positions


def lay_segments(
    rng: random.Random,
    context_length: int,
    segments: Sequence[Sequence[int]],
) -> List[int]:
    context = [random_noise(rng) for _ in range(context_length)]
    positions = choose_insert_positions(rng, context_length, [len(seg) for seg in segments])
    for pos, seg in zip(positions, segments):
        context[pos : pos + len(seg)] = list(seg)
    return context


def generate_passkey_sample(rng: random.Random, context_length: int) -> Sample:
    key_id = rng.randrange(NUM_KEYS)
    answer_id = rng.randrange(NUM_VALUES)
    distractors = []
    for _ in range(3):
        d_key = rng.randrange(NUM_KEYS)
        d_value = rng.randrange(NUM_VALUES)
        distractors.append([PASS, key_token(d_key), answer_token(d_value)])
    target = [PASS, key_token(key_id), answer_token(answer_id)]
    context = lay_segments(rng, context_length, [target, *distractors])
    query = [ASK, key_token(key_id)]
    return Sample("passkey", context, query, answer_id)


def generate_associative_sample(rng: random.Random, context_length: int) -> Sample:
    key_id = rng.randrange(NUM_KEYS)
    answer_id = rng.randrange(NUM_VALUES)
    segments = [[MAP, key_token(key_id), answer_token(answer_id)]]
    used_keys = {key_id}
    for _ in range(4):
        d_key = rng.randrange(NUM_KEYS)
        while d_key in used_keys:
            d_key = rng.randrange(NUM_KEYS)
        used_keys.add(d_key)
        d_value = rng.randrange(NUM_VALUES)
        segments.append([MAP, key_token(d_key), answer_token(d_value)])
    context = lay_segments(rng, context_length, segments)
    query = [ASK, alias_for_key(key_id)]
    return Sample("associative", context, query, answer_id)


def generate_sequential_sample(rng: random.Random, context_length: int) -> Sample:
    key_a = rng.randrange(NUM_KEYS)
    key_b = rng.randrange(NUM_KEYS)
    bridge = rng.randrange(NUM_KEYS)
    answer_id = rng.randrange(NUM_VALUES)
    chain = [
        [STEP, key_token(key_a), key_token(key_b)],
        [STEP, key_token(key_b), bridge_token(bridge)],
        [MAP, bridge_token(bridge), answer_token(answer_id)],
    ]
    distractors = []
    for _ in range(4):
        left_type = rng.choice(["key", "bridge"])
        if left_type == "key":
            left = key_token(rng.randrange(NUM_KEYS))
        else:
            left = bridge_token(rng.randrange(NUM_KEYS))
        distractors.append([rng.choice([MAP, STEP]), left, answer_token(rng.randrange(NUM_VALUES))])
    context = lay_segments(rng, context_length, [*chain, *distractors])
    query = [ASK, alias_for_key(key_a)]
    return Sample("sequential", context, query, answer_id)


def generate_compositional_sample(rng: random.Random, context_length: int) -> Sample:
    key_a = rng.randrange(NUM_KEYS)
    key_b = rng.randrange(NUM_KEYS)
    bridge_c = rng.randrange(NUM_KEYS)
    bridge_d = rng.randrange(NUM_KEYS)
    answer_id = rng.randrange(NUM_VALUES)
    chain = [
        [STEP, key_token(key_a), key_token(key_b)],
        [STEP, key_token(key_b), bridge_token(bridge_c)],
        [STEP, bridge_token(bridge_c), bridge_token(bridge_d)],
        [MAP, bridge_token(bridge_d), answer_token(answer_id)],
    ]
    distractors = []
    for _ in range(8):
        left_type = rng.choice(["key", "bridge"])
        right_type = rng.choice(["key", "bridge", "value"])
        left = key_token(rng.randrange(NUM_KEYS)) if left_type == "key" else bridge_token(rng.randrange(NUM_KEYS))
        if right_type == "key":
            right = key_token(rng.randrange(NUM_KEYS))
            marker = STEP
        elif right_type == "bridge":
            right = bridge_token(rng.randrange(NUM_KEYS))
            marker = STEP
        else:
            right = answer_token(rng.randrange(NUM_VALUES))
            marker = MAP
        distractors.append([marker, left, right])
    context = lay_segments(rng, context_length, [*chain, *distractors])
    query = [ASK, alias_for_key(key_a)]
    return Sample("compositional", context, query, answer_id)


TASK_GENERATORS = {
    "passkey": generate_passkey_sample,
    "associative": generate_associative_sample,
    "sequential": generate_sequential_sample,
    "compositional": generate_compositional_sample,
}


class LongContextDataset(Dataset):
    """Generates deterministic synthetic samples on the fly."""

    def __init__(
        self,
        size: int,
        context_length: int,
        tasks: Sequence[str],
        seed: int,
    ) -> None:
        self.size = size
        self.context_length = context_length
        self.tasks = list(tasks)
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        rng = random.Random(self.seed + index)
        task = self.tasks[index % len(self.tasks)]
        sample = TASK_GENERATORS[task](rng, self.context_length)
        sequence = sample.context_tokens + [SEP] + sample.query_tokens
        return {
            "task": sample.task,
            "context_tokens": torch.tensor(sample.context_tokens, dtype=torch.long),
            "query_tokens": torch.tensor(sample.query_tokens, dtype=torch.long),
            "sequence_tokens": torch.tensor(sequence, dtype=torch.long),
            "answer": torch.tensor(sample.answer_id, dtype=torch.long),
        }


def collate_batch(batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    def pad_stack(key: str) -> torch.Tensor:
        tensors = [item[key] for item in batch]
        max_len = max(t.size(0) for t in tensors)
        padded = torch.full((len(tensors), max_len), PAD, dtype=torch.long)
        lengths = []
        for row, tensor in enumerate(tensors):
            padded[row, : tensor.size(0)] = tensor
            lengths.append(tensor.size(0))
        return padded, torch.tensor(lengths, dtype=torch.long)

    context_tokens, context_lengths = pad_stack("context_tokens")
    query_tokens, query_lengths = pad_stack("query_tokens")
    sequence_tokens, sequence_lengths = pad_stack("sequence_tokens")
    answers = torch.stack([item["answer"] for item in batch])
    tasks = [str(item["task"]) for item in batch]
    return {
        "tasks": tasks,
        "context_tokens": context_tokens,
        "context_lengths": context_lengths,
        "query_tokens": query_tokens,
        "query_lengths": query_lengths,
        "sequence_tokens": sequence_tokens,
        "sequence_lengths": sequence_lengths,
        "answer": answers,
    }


def task_names() -> Iterable[str]:
    return TASK_GENERATORS.keys()


def text_symbol_index(kind: str, idx: int) -> int:
    if kind == "key":
        return idx
    if kind in {"alias", "bridge"}:
        return NUM_KEYS + idx if kind == "bridge" else idx
    raise ValueError(f"Unsupported symbol kind: {kind}")


def build_graph_targets(events: Sequence[tuple[str, int, int, bool]]) -> tuple[torch.Tensor, torch.Tensor]:
    num_symbols = NUM_KEYS * 2
    map_target = torch.zeros(num_symbols, NUM_VALUES, dtype=torch.float32)
    step_target = torch.zeros(num_symbols, num_symbols, dtype=torch.float32)
    for kind, source_idx, target_idx, is_value in events:
        if kind == "map" and is_value:
            map_target[source_idx, target_idx] = 1.0
        elif kind == "step" and not is_value:
            step_target[source_idx, target_idx] = 1.0
    return map_target, step_target


def render_noise_sentence(rng: random.Random) -> List[int]:
    return [text_noise_word(rng), TEXT_AND, text_noise_word(rng), TEXT_PERIOD]


def render_map_sentence(source_kind: str, source_idx: int, value_idx: int) -> tuple[List[int], tuple[str, int, int, bool]]:
    if source_kind == "key":
        prefix = [TEXT_ENTITY, text_key_name(source_idx)]
        source_symbol = source_idx
    elif source_kind == "bridge":
        prefix = [TEXT_BRIDGE, text_bridge_name(source_idx)]
        source_symbol = NUM_KEYS + source_idx
    else:
        raise ValueError(f"Unsupported map source kind: {source_kind}")
    sentence = [*prefix, TEXT_MAPS, TEXT_TO, TEXT_VALUE, text_value_name(value_idx), TEXT_PERIOD]
    return sentence, ("map", source_symbol, value_idx, True)


def render_alias_sentence(alias_idx: int, key_idx: int) -> tuple[List[int], tuple[str, int, int, bool]]:
    sentence = [TEXT_ALIAS, text_alias_name(alias_idx), TEXT_REFERS, TEXT_TO, TEXT_ENTITY, text_key_name(key_idx), TEXT_PERIOD]
    return sentence, ("step", alias_idx, key_idx, False)


def render_link_sentence(source_kind: str, source_idx: int, target_kind: str, target_idx: int) -> tuple[List[int], tuple[str, int, int, bool]]:
    if source_kind == "key":
        left = [TEXT_ENTITY, text_key_name(source_idx)]
        source_symbol = source_idx
    elif source_kind == "bridge":
        left = [TEXT_BRIDGE, text_bridge_name(source_idx)]
        source_symbol = NUM_KEYS + source_idx
    else:
        raise ValueError(f"Unsupported link source kind: {source_kind}")

    if target_kind == "key":
        right = [TEXT_ENTITY, text_key_name(target_idx)]
        target_symbol = target_idx
    elif target_kind == "bridge":
        right = [TEXT_BRIDGE, text_bridge_name(target_idx)]
        target_symbol = NUM_KEYS + target_idx
    else:
        raise ValueError(f"Unsupported link target kind: {target_kind}")

    sentence = [*left, TEXT_LINKS, TEXT_TO, *right, TEXT_PERIOD]
    return sentence, ("step", source_symbol, target_symbol, False)


def render_question(source_kind: str, source_idx: int) -> List[int]:
    if source_kind == "alias":
        source = [TEXT_ALIAS, text_alias_name(source_idx)]
    elif source_kind == "key":
        source = [TEXT_ENTITY, text_key_name(source_idx)]
    elif source_kind == "bridge":
        source = [TEXT_BRIDGE, text_bridge_name(source_idx)]
    else:
        raise ValueError(f"Unsupported question source kind: {source_kind}")
    return [TEXT_QUESTION, TEXT_WHAT, TEXT_IS, TEXT_VALUE, TEXT_OF, *source, TEXT_QMARK, TEXT_ANSWER]


def flatten_sentences(sentences: Sequence[Sequence[int]]) -> List[int]:
    flat: List[int] = []
    for sentence in sentences:
        flat.extend(sentence)
    return flat


def generate_text_graph_sample(rng: random.Random, task: str) -> TextGraphSample:
    sentences: List[List[int]] = []
    events: List[tuple[str, int, int, bool]] = []

    if task == "passkey":
        key_id = rng.randrange(NUM_KEYS)
        value_id = rng.randrange(NUM_VALUES)
        for _ in range(4):
            d_key = rng.randrange(NUM_KEYS)
            d_val = rng.randrange(NUM_VALUES)
            sent, event = render_map_sentence("key", d_key, d_val)
            sentences.append(sent)
            events.append(event)
        question_kind = "key"
        question_idx = key_id
        answer_value = value_id
        sent, event = render_map_sentence("key", key_id, value_id)
        sentences.append(sent)
        events.append(event)
    elif task == "associative":
        key_id = rng.randrange(NUM_KEYS)
        value_id = rng.randrange(NUM_VALUES)
        alias_id = key_id
        sent, event = render_alias_sentence(alias_id, key_id)
        sentences.append(sent)
        events.append(event)
        sent, event = render_map_sentence("key", key_id, value_id)
        sentences.append(sent)
        events.append(event)
        for _ in range(4):
            d_key = rng.randrange(NUM_KEYS)
            d_val = rng.randrange(NUM_VALUES)
            sent, event = render_map_sentence("key", d_key, d_val)
            sentences.append(sent)
            events.append(event)
        question_kind = "alias"
        question_idx = alias_id
        answer_value = value_id
    elif task == "sequential":
        key_a = rng.randrange(NUM_KEYS)
        key_b = rng.randrange(NUM_KEYS)
        bridge = rng.randrange(NUM_KEYS)
        value_id = rng.randrange(NUM_VALUES)
        alias_id = key_a
        chain = [
            render_alias_sentence(alias_id, key_a),
            render_link_sentence("key", key_a, "key", key_b),
            render_link_sentence("key", key_b, "bridge", bridge),
            render_map_sentence("bridge", bridge, value_id),
        ]
        for sent, event in chain:
            sentences.append(sent)
            events.append(event)
        for _ in range(5):
            if rng.random() < 0.5:
                sent, event = render_link_sentence("key", rng.randrange(NUM_KEYS), "bridge", rng.randrange(NUM_KEYS))
            else:
                sent, event = render_map_sentence("bridge", rng.randrange(NUM_KEYS), rng.randrange(NUM_VALUES))
            sentences.append(sent)
            events.append(event)
        question_kind = "alias"
        question_idx = alias_id
        answer_value = value_id
    elif task == "compositional":
        key_a = rng.randrange(NUM_KEYS)
        key_b = rng.randrange(NUM_KEYS)
        bridge_c = rng.randrange(NUM_KEYS)
        bridge_d = rng.randrange(NUM_KEYS)
        value_id = rng.randrange(NUM_VALUES)
        alias_id = key_a
        chain = [
            render_alias_sentence(alias_id, key_a),
            render_link_sentence("key", key_a, "key", key_b),
            render_link_sentence("key", key_b, "bridge", bridge_c),
            render_link_sentence("bridge", bridge_c, "bridge", bridge_d),
            render_map_sentence("bridge", bridge_d, value_id),
        ]
        for sent, event in chain:
            sentences.append(sent)
            events.append(event)
        for _ in range(8):
            roll = rng.random()
            if roll < 0.33:
                sent, event = render_link_sentence("key", rng.randrange(NUM_KEYS), "key", rng.randrange(NUM_KEYS))
            elif roll < 0.66:
                sent, event = render_link_sentence("bridge", rng.randrange(NUM_KEYS), "bridge", rng.randrange(NUM_KEYS))
            else:
                sent, event = render_map_sentence("bridge", rng.randrange(NUM_KEYS), rng.randrange(NUM_VALUES))
            sentences.append(sent)
            events.append(event)
        question_kind = "alias"
        question_idx = alias_id
        answer_value = value_id
    else:
        raise ValueError(f"Unsupported text-graph task: {task}")

    for _ in range(6):
        sentences.append(render_noise_sentence(rng))
    rng.shuffle(sentences)

    question = render_question(question_kind, question_idx)
    full_sequence = [TEXT_BOS, *flatten_sentences(sentences), *question, text_value_name(answer_value), TEXT_PERIOD, TEXT_EOS]
    answer_position = len(full_sequence) - 3
    map_target, step_target = build_graph_targets(events)
    return TextGraphSample(
        task=task,
        sequence_tokens=full_sequence,
        answer_token=text_value_name(answer_value),
        answer_position=answer_position,
        map_target=map_target,
        step_target=step_target,
    )


class TextGraphDataset(Dataset):
    """Synthetic raw-text corpus with graph supervision."""

    def __init__(self, size: int, tasks: Sequence[str], seed: int) -> None:
        self.size = size
        self.tasks = list(tasks)
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        rng = random.Random(self.seed + index)
        task = self.tasks[index % len(self.tasks)]
        sample = generate_text_graph_sample(rng, task)
        input_tokens = torch.tensor(sample.sequence_tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(sample.sequence_tokens[1:], dtype=torch.long)
        return {
            "task": sample.task,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "answer_token": torch.tensor(sample.answer_token, dtype=torch.long),
            "answer_position": torch.tensor(sample.answer_position - 1, dtype=torch.long),
            "map_target": sample.map_target,
            "step_target": sample.step_target,
        }


def collate_text_graph_batch(batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    max_len = max(item["input_tokens"].size(0) for item in batch)
    input_tokens = torch.full((len(batch), max_len), TEXT_PAD, dtype=torch.long)
    target_tokens = torch.full((len(batch), max_len), TEXT_PAD, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for row, item in enumerate(batch):
        seq_len = item["input_tokens"].size(0)
        input_tokens[row, :seq_len] = item["input_tokens"]
        target_tokens[row, :seq_len] = item["target_tokens"]
        lengths[row] = seq_len
    return {
        "tasks": [str(item["task"]) for item in batch],
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "lengths": lengths,
        "answer_token": torch.stack([item["answer_token"] for item in batch]),
        "answer_position": torch.stack([item["answer_position"] for item in batch]),
        "map_target": torch.stack([item["map_target"] for item in batch]),
        "step_target": torch.stack([item["step_target"] for item in batch]),
    }
