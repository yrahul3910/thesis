import subprocess

import glob


for file in glob.glob('runs-*.txt'):
    _ = subprocess.Popen(f'grep Running {file} | wc -l', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode('utf-8')
    count = int(_.split()[0])
    print(f'{file}: {round(count / (768) * 100, 2)}%')
