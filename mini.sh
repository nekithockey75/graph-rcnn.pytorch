# Download pre-processed dataset
mkdir -p datasets/vg_bm

# Download data
./download.sh

# Move data
mv mini_imdb_1024.h5 datasets/vg_bm/imdb_1024.h5
mv mini_proposals.h5 datasets/vg_bm/proposals.h5
mv mini_VG-SGG.h5 datasets/vg_bm/VG-SGG.h5
mv mini_VG-SGG-dicts.json datasets/vg_bm/VG-SGG-dicts.json