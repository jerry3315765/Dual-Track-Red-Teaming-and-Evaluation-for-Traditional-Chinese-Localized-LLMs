import pandas as pd
import os
from typing import Dict, List, Any, Optional
from collections import Counter


class DataLoader:
    """
    DataLoader class for loading and managing different jailbreak datasets.

    This class provides a unified interface to load various jailbreak datasets
    and perform statistical analysis on them.
    """

    def __init__(self, data_dir: str = "data/datasets"):
        """
        Initialize the DataLoader.

        Args:
            data_dir (str): Base directory path where datasets are stored
        """
        self.data_dir = data_dir
        self.loaded_data = {}
        self.dataset_stats = {}

    def load(
        self, dataset_name: str, dataset_size: str = "large", **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Unified function to load different datasets.

        Args:
            dataset_name (str): Name of the dataset to load (e.g., 'strongreject', 'ckjail',
                           'jbb_behaviors_cka', 'strongreject_cka')
            dataset_size (str): Size of the dataset ('large' or 'small') for datasets that support it
            **kwargs: Additional arguments for specific dataset loaders

        Returns:
            List[Dict[str, Any]]: List of loaded data samples

        Raises:
            ValueError: If dataset_name is not supported
        """
        name = dataset_name.lower()
        if name == "strongreject":
            data = self._load_strongreject(dataset_size, **kwargs)
            return data
        elif name == "ckjail":
            # ckjail has a single CSV; dataset_size is ignored
            data = self._load_ckjail(**kwargs)
            return data
        elif name == "jbb_behaviors_cka" or name == "jbb-behaviors-cka":
            # Load JBB-Behaviors CKA dataset (Subset of JBB-Behaviors)
            data = self._load_jbb_behaviors_cka(**kwargs)
            return data
        elif name == "strongreject_cka" or name == "strongreject-cka":
            # Load StrongReject CKA dataset (Subset of StrongReject)
            data = self._load_strongreject_cka(**kwargs)
            return data
        elif name == "harmbench_cka" or name == "harmbench-cka" or name == "harmbench":
            # Load HarmBench CKA dataset
            data = self._load_harmbench_cka(**kwargs)
            return data
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' is not supported yet. "
                f"Supported datasets: ['strongreject', 'ckjail', 'jbb_behaviors_cka', 'strongreject_cka', 'harmbench_cka']"
            )

    def _load_jbb_behaviors_cka(self) -> List[Dict[str, Any]]:
        """
        Load JBB-Behaviors CKA dataset.

        This dataset is located in the cka-attack subdirectory and follows
        the same CSV format as CKJail dataset.

        Expects CSV at <data_dir>/cka-attack/jbb-behaviors_cka_dataset.csv with columns:
        - category
        - source
        - forbidden_prompt

        Returns:
            List[Dict[str, Any]]: List of data samples with keys:
                                - 'query': forbidden_prompt content
                                - 'source': source information
                                - 'category': category classification

        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = os.path.join(
            self.data_dir, "cka-attack", "jbb-behaviors_cka_dataset.csv"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        # Validate required columns
        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert DataFrame to list of dictionaries
        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            # Preserve target_str if it exists in the dataset
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        # Store loaded data for statistics
        dataset_key = "jbb_behaviors_cka"
        self.loaded_data[dataset_key] = data_list
        self._compute_statistics(dataset_key, data_list)

        print(
            f"Successfully loaded {len(data_list)} samples from JBB-Behaviors CKA dataset"
        )
        return data_list

    def _load_strongreject_cka(self) -> List[Dict[str, Any]]:
        """
        Load StrongReject CKA dataset.

        This dataset is located in the cka-attack subdirectory and follows
        the same CSV format as CKJail dataset.

        Expects CSV at <data_dir>/cka-attack/strongreject_cka_dataset.csv with columns:
        - category
        - source
        - forbidden_prompt

        Returns:
            List[Dict[str, Any]]: List of data samples with keys:
                                - 'query': forbidden_prompt content
                                - 'source': source information
                                - 'category': category classification

        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = os.path.join(
            self.data_dir, "cka-attack", "strongreject_cka_dataset.csv"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        # Validate required columns
        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert DataFrame to list of dictionaries
        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            # Preserve target_str if it exists in the dataset
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        # Store loaded data for statistics
        dataset_key = "strongreject_cka"
        self.loaded_data[dataset_key] = data_list
        self._compute_statistics(dataset_key, data_list)

        print(
            f"Successfully loaded {len(data_list)} samples from StrongReject CKA dataset"
        )
        return data_list

    def _load_harmbench_cka(self) -> List[Dict[str, Any]]:
        """
        Load HarmBench CKA dataset.

        This dataset is located in the cka-attack subdirectory and follows
        the same CSV format as other CKA datasets.

        Expects CSV at <data_dir>/cka-attack/harmbench_cka_dataset.csv with columns:
        - category
        - source
        - forbidden_prompt
        - target_str (optional, will be included if present)

        Returns:
            List[Dict[str, Any]]: List of data samples with keys:
                                - 'query': forbidden_prompt content
                                - 'source': source information
                                - 'category': category classification
                                - 'target_str': (optional) target string if present in dataset

        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = os.path.join(
            self.data_dir, "cka-attack", "harmbench_cka_dataset.csv"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        # Validate required columns
        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert DataFrame to list of dictionaries
        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            # Preserve target_str if it exists in the dataset
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        # Store loaded data for statistics
        dataset_key = "harmbench_cka"
        self.loaded_data[dataset_key] = data_list
        self._compute_statistics(dataset_key, data_list)

        print(
            f"Successfully loaded {len(data_list)} samples from HarmBench CKA dataset"
        )
        return data_list

    def _load_strongreject(self, dataset_size: str = "large") -> List[Dict[str, Any]]:
        """
        Load StrongReject dataset.

        Args:
            dataset_size (str): Size of dataset to load ('large' or 'small')
                              'large' loads strongreject_dataset.csv
                              'small' loads strongreject_small_dataset.csv

        Returns:
            List[Dict[str, Any]]: List of data samples with keys:
                                 - 'query': forbidden_prompt content
                                 - 'source': source information
                                 - 'category': category classification

        Raises:
            FileNotFoundError: If the specified dataset file doesn't exist
            ValueError: If dataset_size is not 'large' or 'small'
        """
        # Validate dataset_size parameter
        if dataset_size not in ["large", "small"]:
            raise ValueError("dataset_size must be 'large' or 'small'")

        # Determine filename based on dataset size
        if dataset_size == "large":
            filename = "strongreject_dataset.csv"
        else:  # dataset_size == "small"
            filename = "strongreject_small_dataset.csv"

        # Construct full file path
        file_path = os.path.join(self.data_dir, "strongreject", filename)

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        # Validate required columns
        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert DataFrame to list of dictionaries with renamed keys
        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            # If the target_str column exists in the dataset, keep it in the sample for use by the pair method.
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        # Store loaded data for statistics
        dataset_key = f"strongreject_{dataset_size}"
        self.loaded_data[dataset_key] = data_list

        # Compute and store statistics
        self._compute_statistics(dataset_key, data_list)

        print(
            f"Successfully loaded {len(data_list)} samples from StrongReject ({dataset_size}) dataset"
        )

        return data_list

    def _load_ckjail(self) -> List[Dict[str, Any]]:
        """
        Load CKJail dataset.

        Expects a single CSV at <data_dir>/ckjail/ckjail_dataset.csv with columns:
          - category
          - source
          - forbidden_prompt

        Returns:
            List[Dict[str, Any]]: List of data samples with keys:
                                 - 'query'
                                 - 'source'
                                 - 'category'
        """
        file_path = os.path.join(self.data_dir, "ckjail", "ckjail_dataset.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        dataset_key = "ckjail"
        self.loaded_data[dataset_key] = data_list
        self._compute_statistics(dataset_key, data_list)

        print(f"Successfully loaded {len(data_list)} samples from CKJail dataset")
        return data_list

    def _compute_statistics(self, dataset_key: str, data: List[Dict[str, Any]]) -> None:
        """
        Compute statistics for the loaded dataset.

        Args:
            dataset_key (str): Key to identify the dataset
            data (List[Dict[str, Any]]): Loaded data samples
        """
        if not data:
            self.dataset_stats[dataset_key] = {}
            return

        # Basic statistics
        total_samples = len(data)

        # Category distribution
        categories = [sample["category"] for sample in data]
        category_counts = Counter(categories)

        # Source distribution
        sources = [sample["source"] for sample in data]
        source_counts = Counter(sources)

        # Query length statistics
        query_lengths = [len(sample["query"]) for sample in data]
        avg_query_length = sum(query_lengths) / len(query_lengths)
        min_query_length = min(query_lengths)
        max_query_length = max(query_lengths)

        # Store statistics
        self.dataset_stats[dataset_key] = {
            "total_samples": total_samples,
            "category_distribution": dict(category_counts),
            "source_distribution": dict(source_counts),
            "query_length_stats": {
                "average": round(avg_query_length, 2),
                "minimum": min_query_length,
                "maximum": max_query_length,
            },
        }

    def print_statistics(self, dataset_key: Optional[str] = None) -> None:
        """
        Print statistics for loaded datasets.

        Args:
            dataset_key (str, optional): Specific dataset to print stats for.
                                       If None, prints stats for all loaded datasets.
        """
        if dataset_key:
            # Print statistics for specific dataset
            if dataset_key not in self.dataset_stats:
                print(f"No statistics available for dataset: {dataset_key}")
                return

            self._print_dataset_stats(dataset_key, self.dataset_stats[dataset_key])
        else:
            # Print statistics for all loaded datasets
            if not self.dataset_stats:
                print("No datasets loaded yet.")
                return

            for key, stats in self.dataset_stats.items():
                self._print_dataset_stats(key, stats)
                print("-" * 60)  # Separator between datasets

    def _print_dataset_stats(self, dataset_key: str, stats: Dict[str, Any]) -> None:
        """
        Print statistics for a single dataset.

        Args:
            dataset_key (str): Dataset identifier
            stats (Dict[str, Any]): Statistics dictionary
        """
        print(f"\n=== Dataset Statistics: {dataset_key} ===")
        print(f"Total Samples: {stats.get('total_samples', 0)}")

        # Category distribution
        print(f"\nCategory Distribution:")
        category_dist = stats.get("category_distribution", {})
        for category, count in sorted(category_dist.items()):
            percentage = (
                (count / stats["total_samples"]) * 100
                if stats.get("total_samples", 0)
                else 0.0
            )
            print(f"  {category}: {count} ({percentage:.1f}%)")

        # Source distribution
        print(f"\nSource Distribution:")
        source_dist = stats.get("source_distribution", {})
        for source, count in sorted(source_dist.items()):
            percentage = (
                (count / stats["total_samples"]) * 100
                if stats.get("total_samples", 0)
                else 0.0
            )
            print(f"  {source}: {count} ({percentage:.1f}%)")

        # Query length statistics
        query_stats = stats.get("query_length_stats", {})
        if query_stats:
            print(f"\nQuery Length Statistics:")
            print(f"  Average: {query_stats.get('average', 0)} characters")
            print(f"  Minimum: {query_stats.get('minimum', 0)} characters")
            print(f"  Maximum: {query_stats.get('maximum', 0)} characters")

    def get_loaded_datasets(self) -> List[str]:
        """
        Get list of loaded dataset keys.

        Returns:
            List[str]: List of loaded dataset identifiers
        """
        return list(self.loaded_data.keys())

    def get_dataset(self, dataset_key: str) -> List[Dict[str, Any]]:
        """
        Get loaded dataset by key.

        Args:
            dataset_key (str): Dataset identifier

        Returns:
            List[Dict[str, Any]]: Dataset samples

        Raises:
            KeyError: If dataset_key is not found
        """
        if dataset_key not in self.loaded_data:
            raise KeyError(
                f"Dataset '{dataset_key}' not loaded. "
                f"Available datasets: {list(self.loaded_data.keys())}"
            )

        return self.loaded_data[dataset_key]
