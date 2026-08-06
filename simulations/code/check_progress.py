import os

SUCCESS_LINE = "--> Simulation finished."
REMAINING_OUTFILE = "remaining_dirs.txt"

def is_finished(output_log_path: str) -> bool:
    try:
        with open(output_log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            if not lines:
                return False
            return lines[-1].strip() == SUCCESS_LINE
    except FileNotFoundError:
        return False
    except Exception:
        return False

def last_line(output_log_path: str):
    """Return the last non-empty line of output.log, or None if unavailable."""
    try:
        with open(output_log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        for ln in reversed(lines):
            if ln.strip():
                return ln.strip()
        return None
    except FileNotFoundError:
        return None
    except Exception:
        return None

def main():
    dirs = sorted([d for d in os.listdir(".") if os.path.isdir(d) and d.startswith("x")])

    remaining = []
    cnt = 1
    for d in dirs:
        out = os.path.join(d, "output.log")
        ll = last_line(out)
        if ll is not None:
            print(f"{cnt}. {d}: {ll}")
            cnt += 1
        if not is_finished(out):
            remaining.append(d)

    with open(REMAINING_OUTFILE, "w", encoding="utf-8") as f:
        for d in remaining:
            f.write(d + "\n")

    print(f"Total x* dirs: {len(dirs)}")
    print(f"Remaining (not finished): {len(remaining)}")
    print(f"Wrote: {REMAINING_OUTFILE}")
    if remaining:
        print("List:", ", ".join(remaining))

if __name__ == "__main__":
    main()