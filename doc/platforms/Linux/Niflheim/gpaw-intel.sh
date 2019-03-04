export GPAW_TOOLCHAIN=intel

if [ -z $GPAW ]; then
    GPAW=~/gpaw
fi
if [ -z $ASE ]; then
    ASE=~/ase
fi

if [[ -z $MYPYTHON ]]; then
    MYPYTHON=${GPAW_TOOLCHAIN}-2018b-Python-3.6.6
fi

# Load libxc unless another version is already loaded
if [[ -z "$EBROOTLIBXC" ]]; then
   module load libxc/3.0.1-${GPAW_TOOLCHAIN}-2018b
fi

# Load Python and matplotlib
module load matplotlib/3.0.0-$MYPYTHON

# Load default setups, unless GPAW_SETUP_PATH is already set
if [[ -z $GPAW_SETUP_PATH ]]; then
    module load GPAW-setups
fi

export GPAW_MPI_OPTIONS=""
PLATFORM=linux-x86_64-$CPU_ARCH-el7-3.6
export PATH=$GPAW/tools:$GPAW/build/bin.$PLATFORM:$PATH
export PYTHONPATH=$GPAW:$GPAW/build/lib.$PLATFORM:$PYTHONPATH
export PATH=$ASE/bin:$PATH
export PYTHONPATH=$ASE:$PYTHONPATH
