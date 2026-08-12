"""
Dataset module for multi-cell type fiber mixing and evaluation.
Inherits from fiber_data_iterator in data_utils.py.
"""

import random
import torch
import numpy as np

from data_utils import fiber_data_iterator


class MixedCellFiberDataset(fiber_data_iterator):
    """
    Evaluation dataset that samples fibers from multiple cell types at a shared genomic locus,
    concatenates them into a synthetic mixed fiber stack, and provides per-cell-type ground-truth
    bulk targets for deconvolution evaluation.
    """

    def __init__(self, metadata, fibers_per_entry, context_length,
                 input_flags, num_sample_ccres=1000, seed=919,
                 return_dna=False, bulk_name="N/A"):
        """
        Args:
            metadata (dict): Dataset configuration containing paths and cell types.
            fibers_per_entry (int): Total number of fibers in the combined stack.
            context_length (int): Window length in base pairs.
            input_flags (list): 5-bit feature mask (e.g., [1, 1, 1, 1, 1]).
            num_sample_ccres (int): Number of composite samples generated per epoch.
            cell_ratios (dict): Optional dict mapping cell_type_name -> ratio (e.g., {'GM12878': 0.5, 'K562': 0.5}).
            mode (str): 'val' or 'test' chromosome partition selection.
            seed (int): Random seed for reproducible locus & fiber sampling.
            return_dna (bool): Whether to include sequence tensors.
        """
        super().__init__(
            metadata=metadata,
            fibers_per_entry=fibers_per_entry,
            context_length=context_length,
            iters_per_epoch=num_sample_ccres,
            input_flags=input_flags,
            mode="val",
            seed=seed,
            return_dna=return_dna,
            bulk_name=bulk_name
        )

        try:
            meta_cell_ratios = {cell_type:metadata["cell_types"][cell_type]["ratio"] for cell_type in metadata["cell_types"].keys()}
        except:
            print("Failed to extract cell ratio from meta. Falling back to uniform distribution")
            meta_cell_ratios = None
        self.cell_ratios = self._setup_mixing_ratios(meta_cell_ratios)
        self.fiber_counts_per_cell = self._calculate_fiber_counts()
        self.bulk_name = bulk_name
        self.input_flags = input_flags
        self.num_sample_ccres = num_sample_ccres

    # -------------------------------------------------------------------------
    # Helper 1: Configuration & Ratio Allocation
    # -------------------------------------------------------------------------

    def _setup_mixing_ratios(self, cell_ratios):
        """Normalizes user-specified ratios or defaults to equal proportions."""
        if cell_ratios is None:
            uniform_p = 1.0 / self.num_cell_types
            return {ct: uniform_p for ct in self.cell_type_names}

        total_p = sum(cell_ratios.values())
        return {ct: cell_ratios[ct] / total_p for ct in self.cell_type_names}

    def _calculate_fiber_counts(self):
        """Splits total fibers_per_entry among cell types according to self.cell_ratios."""
        counts = {}
        allocated = 0
        sorted_cts = sorted(self.cell_type_names)

        for i, ct in enumerate(sorted_cts):
            if i == len(sorted_cts) - 1:
                counts[ct] = self.fibers_per_entry - allocated
            else:
                c = int(round(self.cell_ratios[ct] * self.fibers_per_entry))
                counts[ct] = c
                allocated += c

        return counts

    # -------------------------------------------------------------------------
    # Helper 2: Extract Data for a Single Cell Type at a Locus
    # -------------------------------------------------------------------------

    def _sample_single_cell_type(self, cell_idx, ct, locus):
        """Fetches and validates fiber, bulk, and optional DNA data for one cell type."""
        n_fibers_needed = self.fiber_counts_per_cell[ct]
        if n_fibers_needed == 0:
            return None

        # Fetch fibers using inherited method
        fiber_tensor, fiber_dna_tensor, n_fibers = self.get_fiber_data(
            cell_idx, *locus, min_overlap=self.context_length // 8
        )

        # Insufficient reads check
        if n_fibers < min(5, n_fibers_needed):
            return None

        # Fetch BigWig target profile
        bw_tensor = self.get_other_bw_data(cell_idx, *locus)
        if torch.isnan(bw_tensor).any().item():
            return None

        # Trim fiber tensors to exact retrieved count
        trimmed_fiber_tensor = fiber_tensor[:, :, :n_fibers_needed]
        trimmed_dna_tensor = (
            fiber_dna_tensor[:, :, :n_fibers_needed]
            if (self.return_dna and fiber_dna_tensor is not None)
            else None
        )

        return {
            "cell_type": ct,
            "fiber_features": trimmed_fiber_tensor,
            "target_bulk": bw_tensor,
            "fiber_dna": trimmed_dna_tensor,
            "n_fibers": n_fibers_needed
        }

    # -------------------------------------------------------------------------
    # Helper 3: Assemble Mixed Payload Across All Cell Types
    # -------------------------------------------------------------------------

    def _build_composite_sample(self, locus, cell_samples):
        """
        Combines individual cell-type samples into unified tensors,
        creates boolean masks, and computes composite targets.
        """
        mixed_fiber_tensors = np.zeros((len(self.input_features), self.context_length, self.fibers_per_entry), dtype=np.float32)
        mixed_dna_tensors = np.zeros((4, self.context_length, self.fibers_per_entry), dtype=np.float32) if self.return_dna else None
        cell_type_masks = {}
        individual_bulk_targets = {}

        current_fiber_offset = 0

        for sample in cell_samples:
            ct = sample["cell_type"]
            n_fibers = sample["n_fibers"]

            individual_bulk_targets[ct] = sample["target_bulk"]
            mixed_fiber_tensors[:,:,current_fiber_offset:current_fiber_offset+n_fibers] = sample["fiber_features"][:n_fibers]

            if sample["fiber_dna"] is not None:
                mixed_dna_tensors.append(sample["fiber_dna"])
                mixed_dna_tensors[:,:,current_fiber_offset:current_fiber_offset+n_fibers] = sample["fiber_dna"][:n_fibers]

            # Build boolean cell-type mask
            mask = torch.zeros(self.fibers_per_entry, dtype=torch.bool)
            mask[current_fiber_offset : current_fiber_offset + n_fibers] = True
            cell_type_masks[ct] = mask

            current_fiber_offset += n_fibers

        composite_bulk_list = [self.cell_ratios[ct]*individual_bulk_targets[ct] for ct in individual_bulk_targets.keys()]
        composite_bulk = torch.stack(composite_bulk_list, dim=0).sum(dim=0)

        out_dict = {
            "fiber_features": mixed_fiber_tensors,
            "target_bulk": composite_bulk,
            "cell_type_targets": individual_bulk_targets,
            "cell_type_masks": cell_type_masks,
            "n_fibers": current_fiber_offset,
            "locus": locus,
            "mixing_ratios": self.cell_ratios
        }

        if self.return_dna:
            out_dict["genomic_dna"] = self.onehot_for_locus(locus)
            out_dict["fiber_dna"] = mixed_dna_tensors

        return out_dict

    # -------------------------------------------------------------------------
    # Core Generator Loop
    # -------------------------------------------------------------------------

    def __iter__(self):
        self.init_worker_resources()

        ccre_iter_list = self.ccre_list if self.num_sample_ccres == -1 else self.ccre_list[:self.num_sample_ccres]

        for ccre_locus in ccre_iter_list:
            found_valid_locus = False
            out_dict = None

            while not found_valid_locus:
                selected_locus = self.expand_ccre_locus(*ccre_locus)
                cell_samples = []
                failed_sampling = False

                for cell_idx, ct in enumerate(self.cell_type_names):
                    sample = self._sample_single_cell_type(cell_idx, ct, selected_locus)
                    if sample is None and self.fiber_counts_per_cell[ct] > 0:
                        failed_sampling = True
                        break
                    if sample is not None:
                        cell_samples.append(sample)

                if failed_sampling or not cell_samples:
                    continue

                out_dict = self._build_composite_sample(selected_locus, cell_samples)
                found_valid_locus = True

            yield out_dict


#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    import yaml

    config_path = "./configs/evals/eval00.yaml"
    with open(config_path, "r") as f:
        eval_config =  yaml.safe_load(f)

    kwargs = {
        "metadata": eval_config["metadata"],
        "context_length": eval_config["context_length"],
        "fibers_per_entry": eval_config["fibers_per_entry"],
        "num_sample_ccres": eval_config["num_sample_ccres"],
        "bulk_name": eval_config["bulk_name"],
        "seed": eval_config["seed"],
        "input_flags": [1, 1, 1, 1, 1],
        "return_dna": False
    }

    t_set = MixedCellFiberDataset(**kwargs)

    # Single-instance check using dictionary unpacking
    sample_dict = next(iter(t_set))
    print(f"Single instance check -> Locus: {sample_dict['locus']}, Total Fibers: {sample_dict['n_fibers']}")
    print(f"Keys present in dictionary (return_dna={kwargs['return_dna']}): {list(sample_dict.keys())}")
    for ct, mask in sample_dict['cell_type_masks'].items():
        print(f"  - Cell type '{ct}' active fibers count: {mask.sum().item()}")

    # Loop inspection
    print("\n--- Loop Inspection ---")
    for i, batch in enumerate(t_set):
        print(f"Sample {i+1} | Locus: {batch['locus']} | Total Fibers: {batch['n_fibers']} | Composite Target Shape: {batch['target_bulk'].shape}")

    print("All done")


if __name__ == "__main__":
    tester()
