#!/bin/bash

files=(2011_09_26_calib.zip
2011_09_26_drive_0001
2011_09_26_drive_0002
2011_09_26_drive_0005
2011_09_26_drive_0009
2011_09_26_drive_0011
2011_09_26_drive_0015
2011_09_26_drive_0020
2011_09_26_drive_0022
2011_09_26_drive_0028
2011_09_26_drive_0032
2011_09_26_drive_0035
2011_09_26_drive_0046
2011_09_26_drive_0048
2011_09_26_drive_0052
2011_09_26_drive_0059
2011_09_26_drive_0079
2011_09_26_drive_0106
2011_09_28_calib.zip
2011_09_28_drive_0016
2011_09_28_drive_0037
2011_09_28_drive_0038
2011_09_28_drive_0043
2011_09_28_drive_0045
2011_09_29_calib.zip
2011_09_29_drive_0071
)


for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                shortname=$i'_sync.zip'
                fullname=$i'/'$i'_sync.zip'
        else
                shortname=$i
                fullname=$i
        fi
	echo "Downloading: "$shortname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
        unzip -o $shortname
        rm $shortname
done
