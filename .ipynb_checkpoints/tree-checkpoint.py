import os

for root, dirs, files in os.walk('.'):
    level = root.count(os.sep)
    print('  ' * level + os.path.basename(root) + '/')
    for f in files:
        print('  ' * (level + 1) + f)