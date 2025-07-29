import os
import argparse
import random

def main(data_root, test_person):
    """
    Prepares MPIIFaceGaze label files for leave-one-out cross-validation.

    This script combines the label files of all participants except the test person
    into a single 'train.label' file.

    Args:
        data_root (str): The root directory of the preprocessed MPIIFaceGaze dataset.
        test_person (str): The person ID to hold out for testing (e.g., 'p14').
    """
    label_dir = os.path.join(data_root, 'Label')
    if not os.path.isdir(label_dir):
        print(f"Error: Label directory not found at {label_dir}")
        return

    all_persons = [f'p{i:02d}' for i in range(15)]
    if test_person not in all_persons:
        print(f"Error: Test person '{test_person}' is not a valid ID (p00-p14).")
        return

    train_persons = [p for p in all_persons if p != test_person]

    print(f"Test Person: {test_person}")
    print(f"Training Persons: {train_persons}")

    all_train_lines = []
    # Read header from the first person's file to use in combined file
    header = ""
    with open(os.path.join(label_dir, f'{train_persons[0]}.label'), 'r') as f_in:
        header = f_in.readline()

    # Combine all training files
    for person in train_persons:
        person_label_file = os.path.join(label_dir, f'{person}.label')
        with open(person_label_file, 'r') as f_in:
            all_train_lines.extend(f_in.readlines()[1:]) # Skip header

    random.shuffle(all_train_lines)

    # Create the combined train.label file
    train_label_path = os.path.join(label_dir, 'train.label')
    with open(train_label_path, 'w') as f_out:
        f_out.write(header)
        f_out.writelines(all_train_lines)

    print("-" * 50)
    print(f"Successfully created '{train_label_path}' with {len(all_train_lines)} samples.")
    print("You can now start training using this file.")
    print(f"To evaluate, use '{test_person}' as the split in your config.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MPIIFaceGaze labels for training.")
    parser.add_argument('--data_root', type=str, required=True, help="Path to the preprocessed MPIIFaceGaze directory.")
    parser.add_argument('--test_person', type=str, default='p14', help="Person ID to hold out for the test set.")
    args = parser.parse_args()
    main(args.data_root, args.test_person)
