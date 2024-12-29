#!/usr/bin/env python3

import argparse
import sys

def arpa_add_token(source_arpa, target_path):
    try:
        with open(source_arpa, "r") as read_file, open(target_path, "w") as write_file:
            has_added_eos = False
            for line in read_file:
                if not has_added_eos and "ngram 1=" in line:
                    # Extract the count after 'ngram 1='
                    parts = line.strip().split("=")
                    if len(parts) >= 2 and parts[-1].isdigit():
                        count = int(parts[-1])
                        new_count = count + 1
                        # Replace the old count with the new count
                        new_line = line.replace(f"{count}", f"{new_count}")
                        write_file.write(new_line)
                    else:
                        # If the count is not found or not a digit, write the line as is
                        write_file.write(line)
                elif not has_added_eos and "<s>" in line:
                    write_file.write(line)
                    # Replace the first occurrence of <s> with </s>
                    new_line = line.replace("<s>", "</s>", 1)
                    write_file.write(new_line)
                    has_added_eos = True
                else:
                    write_file.write(line)
        print(f"Successfully processed '{source_arpa}' and saved to '{target_path}'.")
    except FileNotFoundError as e:
        print(f"Error: {e.strerror}. File '{e.filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Add a token to an ARPA file.")
    parser.add_argument("source_arpa", help="Path to the source ARPA file.")
    parser.add_argument("target_path", help="Path to save the modified ARPA file.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    arpa_add_token(args.source_arpa, args.target_path)

if __name__ == "__main__":
    main()
