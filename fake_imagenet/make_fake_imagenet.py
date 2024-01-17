import os
import shutil

import torch

# RUN IN CONTAINER


def main():
    meta_file = "/workspaces/continualUtils/fake_imagenet/meta.bin"
    output_base_dir = "/mnt/datasets/fake_imagenet/"
    sample_image = "/workspaces/continualUtils/fake_imagenet/sample.JPEG"

    # Load meta.bin file
    if torch.load(meta_file):
        meta_data = torch.load(meta_file)
    else:
        raise ValueError(
            "Unable to load meta.bin file. Please check the file integrity."
        )

    shutil.copy(meta_file, output_base_dir)

    # Extract wnids from meta_data
    wnids = meta_data[0].keys()

    # Create splits: train and val
    for split in ["train", "val"]:
        split_dir = os.path.join(output_base_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Create directories and copy sample image for each wnid
        for wnid in wnids:
            class_dir = os.path.join(split_dir, wnid)
            os.makedirs(class_dir, exist_ok=True)

            if sample_image:
                # Copy a single image into each directory
                sample_image_dest = os.path.join(class_dir, "sample_image.JPEG")
                os.system(f"cp {sample_image} {sample_image_dest}")


if __name__ == "__main__":
    main()
