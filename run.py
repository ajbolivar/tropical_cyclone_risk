import os
import shutil
import sys
import importlib

if len(sys.argv) > 3:
    namelist_name = sys.argv[3]
    module = importlib.import_module(f"namelists.{namelist_name}")

    sys.modules['namelist'] = module

import namelist
from scripts import generate_land_masks
from util import compute

print(f"Using namelist module: {namelist.__name__}")

if __name__ == '__main__':
    f_base = '%s/%s/' % (namelist.output_directory, namelist.exp_name)
    os.makedirs(f_base, exist_ok = True)
    print('Saving model output to %s' % f_base)
    sys.stdout.flush()

    if len(sys.argv) > 3:
        dest_namelist = '%s/%s.py' % (f_base, sys.argv[3])
        if not os.path.exists(dest_namelist):
            shutil.copyfile('namelists/%s.py' % sys.argv[3], dest_namelist)
    else:
        dest_namelist = '%s/namelist.py' % f_base
        if not os.path.exists(dest_namelist):
            shutil.copyfile('./namelist.py', dest_namelist)

    generate_land_masks.generate_land_masks()
    
    if namelist.gnu_parallel == True:
        if namelist.seeding == 'manual':
           compute.create_yearly_files(sys.argv[2])
                
        compute.compute_downscaling_inputs(int(sys.argv[2]))
        print('Running tracks for basin %s...' % sys.argv[1])
        sys.stdout.flush()
        
        compute.run_downscaling(sys.argv[1], int(sys.argv[2]))

    else:
        compute.compute_downscaling_inputs()
        print('Running tracks for basin %s...' % sys.argv[1])
        compute.run_downscaling(sys.argv[1])
