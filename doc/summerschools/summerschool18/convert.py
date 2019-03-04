# creates: batteries/batteries1.ipynb
# creates: batteries/batteries2.ipynb
# creates: batteries/batteries3.ipynb
# creates: catalysis/n2_on_metal.ipynb, catalysis/neb.ipynb
# creates: catalysis/vibrations.ipynb
# creates: magnetism/magnetism1.ipynb, magnetism/magnetism2.ipynb
# creates: machinelearning/machinelearning.ipynb
# creates: photovoltaics/pv1.ipynb, photovoltaics/pv2.ipynb
# creates: photovoltaics/pv3.ipynb
import json
from pathlib import Path


def convert(path):
    assert path.name.endswith('.master.ipynb')
    data = json.loads(path.read_text())
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            lines = cell['source']
            for i, line in enumerate(lines):
                if ' # student:' in line:
                    a, b = (x.strip() for x in line.split('# student:'))
                    lines[i] = line.split(a)[0] + b + '\n'
                elif line.lower().startswith('# teacher'):
                    del lines[i:]
                    break
    new = path.with_name(path.name.rsplit('.', 2)[0] + '.ipynb')
    new.write_text(json.dumps(data, indent=1))


for path in Path().glob('*/*.master.ipynb'):
    print(path)
    convert(path)
