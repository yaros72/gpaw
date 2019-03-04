if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

if [ -z $GPAW ]; then
    GPAW=~/gpaw
fi
if [ -z $ASE ]; then
    ASE=~/ase
fi

module load GPAW/1.4.0-foss-2018b-Python-3.6.6
module unload ASE

export GPAW_MPI_OPTIONS=""
PLATFORM=linux-x86_64-$CPU_ARCH-el7-3.6
export PATH=$GPAW/tools:$GPAW/build/bin.$PLATFORM:$PATH
export PYTHONPATH=$GPAW:$GPAW/build/lib.$PLATFORM:$PYTHONPATH
export PATH=$ASE/bin:$PATH
export PYTHONPATH=$ASE:$PYTHONPATH
