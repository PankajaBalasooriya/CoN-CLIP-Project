from .finetuning_datasets import *
from .evaluation_datasets import *


def get_finetuning_dataset(args, preprocess):
	if args.negative_images == "off":
		return FineTuningDataset(args, preprocess)
	elif args.negative_images == "on":
		try:
			return FineTuningDatasetWithNegatives(args, preprocess)
		except TypeError:
			return FineTuningDatasetWithNegatives(transform=preprocess)
	elif args.negative_images == "on+":
		try:
			return FineTuningDatasetWithNegatives(args, preprocess)
		except TypeError:
			return FineTuningDatasetWithNegatives(transform=preprocess)
