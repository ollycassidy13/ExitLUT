import numpy as np
import torch
from torch.utils.data import Dataset


class SoftmaxApproxDataset(Dataset):
	def __init__(
		self,
		dataset_file=None,
		dataset_config=None,
		split="train",
		input_length=16,
		output_length=5,
		train_size=8192,
		valid_size=2048,
		test_size=2048,
		dataset_seed=1234,
		confidence_threshold=0.6,
		logit_scale=128.0,
	):
		super().__init__()
		if not (0.0 < confidence_threshold < 1.0):
			raise ValueError("confidence_threshold must be in (0, 1).")
		if logit_scale <= 0:
			raise ValueError("logit_scale must be > 0")
		if split not in ("train", "valid", "test"):
			raise ValueError(f"Unsupported split '{split}'.")

		if dataset_file is not None:
			npz = np.load(str(dataset_file))
			all_logits = torch.from_numpy(npz["logits"].astype(np.float32))
			all_labels = torch.from_numpy(npz["exit_labels"].astype(np.int64))
			N = all_logits.shape[0]

			perm = torch.randperm(N, generator=torch.Generator().manual_seed(int(dataset_seed)))
			all_logits = all_logits[perm]
			all_labels = all_labels[perm]

			n_train = min(int(train_size), N)
			n_valid = min(int(valid_size), N - n_train)
			n_test  = min(int(test_size),  N - n_train - n_valid)
			starts  = {"train": 0, "valid": n_train, "test": n_train + n_valid}
			sizes   = {"train": n_train, "valid": n_valid, "test": n_test}
			s = starts[split]
			x_raw = all_logits[s : s + sizes[split]]
			self._y = all_labels[s : s + sizes[split]]

			x_int8 = torch.clamp(torch.floor(x_raw), -128, 127).to(torch.int32)
			self._x = x_int8.float() * (float(logit_scale) / 128.0)
		else:
			if input_length <= 1:
				raise ValueError("input_length must be > 1 for softmax confidence.")

			size_by_split = {"train": int(train_size), "valid": int(valid_size), "test": int(test_size)}
			split_offset  = {"train": 0, "valid": 1, "test": 2}[split]
			generator = torch.Generator().manual_seed(int(dataset_seed) + split_offset)

			x_int8 = torch.randint(
				-128, 128,
				(size_by_split[split], input_length),
				dtype=torch.int16,
				generator=generator,
			)
			x_logit = x_int8.to(torch.float32) * (float(logit_scale) / 128.0)
			self._x = x_logit
			probs = torch.softmax(x_logit, dim=1)
			self._y = (torch.max(probs, dim=1).values >= float(confidence_threshold)).long()

	def __len__(self):
		return self._x.shape[0]

	def __getitem__(self, idx):
		return self._x[idx], self._y[idx]
