import re

# Read log
with open("training_output.log", 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Test split
sections = re.split(r'Training with LR = ', content)[1:]
print(f"Found {len(sections)} sections")

for i, section in enumerate(sections):
    # Get first 100 chars
    preview = section[:200].replace('\n', ' ')
    print(f"\nSection {i}: {preview}...")
    
    # Try to extract LR
    lr_match = re.match(r'([\d.]+)', section)
    if lr_match:
        lr = float(lr_match.group(1))
        print(f"  LR = {lr}")
        
        # Try to find epochs
        pattern = r"\d+/\d+ - [\d.]+s[^\n-]*- accuracy: ([\d.]+) - loss: ([\d.]+(?:e[+-]?\d+)?) - val_accuracy: ([\d.]+) - val_loss: ([\d.]+(?:e[+-]?\d+)?)"
        epochs = re.findall(pattern, section, re.MULTILINE)
        print(f"  Found {len(epochs)} epochs")
        if epochs:
            print(f"  First epoch: acc={epochs[0][0]}, loss={epochs[0][1]}, val_acc={epochs[0][2]}, val_loss={epochs[0][3]}")
