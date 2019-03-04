#!/bin/sh

if [[ -z $GPAW_TOOLCHAIN ]]; then
    echo "You should source gpaw's gpaw-intel.sh or gpaw-foss.sh script to select a proper toolchain."
    echo "   Continuing with the foss toolchain for compatibility.  THIS GIVES INFERIOR PERFORMANCE!"
    echo "   See https://wiki.fysik.dtu.dk/gpaw/platforms/Linux/Niflheim/load.html"
    GPAW_TOOLCHAIN=foss
fi

ICMD=""
for INPUT in "$@"
do
    ICMD="$ICMD$INPUT && "
done

nh=doc/platforms/Linux/Niflheim
rm -rf build
cmd="${ICMD} cd $PWD && python setup.py --remove-default-flags build_ext --customize=$nh/el7-${GPAW_TOOLCHAIN}.py"
echo "Compiling on slid: $cmd"
ssh slid "$cmd > broadwell-el7-${GPAW_TOOLCHAIN}.log 2>&1"
mv configuration.log configuration-broadwell-el7-${GPAW_TOOLCHAIN}.log
echo "Compiling on thul: $cmd"
ssh thul "$cmd > sandybridge-el7-${GPAW_TOOLCHAIN}.log 2>&1"
mv configuration.log configuration-sandybridge-el7-${GPAW_TOOLCHAIN}.log
echo "Compiling on fjorm: $cmd"
ssh fjorm "$cmd > nehalem-el7-${GPAW_TOOLCHAIN}.log 2>&1"
mv configuration.log configuration-nehalem-el7-${GPAW_TOOLCHAIN}.log

(cd build && ln -sf bin.linux-x86_64-{sandybridge,ivybridge}-el7-3.6)
(cd build && ln -sf lib.linux-x86_64-{sandybridge,ivybridge}-el7-3.6)
