cd /home/qktttt/mpich

git submodule update --init --recursive

# Needed for Git checkouts if generated files are missing/stale.
# You can skip this only for a release tarball that already has configure ready.
./autogen.sh

./configure --prefix=/home/qktttt/mpich-install \
  --with-pmi=pmi2 --with-pmi2=${CRAY_PMI_PREFIX} --with-pm=no
#./configure --prefix="$HOME/mpich-install" 2>&1 | tee c.txt
make -j 8 2>&1 | tee m.txt
make install 2>&1 | tee mi.txt

export PATH="$HOME/mpich-install/bin:$PATH"
mpichversion
mpiexec -n 2 hostname