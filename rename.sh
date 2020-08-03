#!/bin/bash


# Special cases
mv './60795_Rhodocollybia_butyracea f. asema' './Rhodocollybia_butyracea'
mv './61140_Bolbitius_reticulatus f. aleuriatus' './Bolbitius_reticulatus'
mv './14901_Hygrophorus_hypothejus f. hypothejus ' './Hygrophorus_hypothejus'
mv '61013_Hydnum_umbilicatum ss auct. eur.' './Hydnum_umbilicatum'

mv 65022_Inocybe_rimosa/* 15468_Inocybe_rimosa/
rm -rf 65022_Inocybe_rimosa/
mv 65010_Cortinarius_anomalus/* 12279_Cortinarius_anomalus
rm -rf 65010_Cortinarius_anomalus/

# General cases
rename 's/([^\s]*)\svar\.\s+(.*)/$1_$2/' *
rename 's/[0-9]+_([^\.]*)(\.)?/$1/' *
IFS=$'\n'
for d in $(find -type d -regex "[^ ]* [a-zA-Z]+.*")
do
    mv $d/* $(echo $d | sed -E 's~([^ ]*).*~\1/~')
    rm -r $d
done
rename 's/([^\s]*).*/$1/' *

# ?
rm -rf 'Xylodon_flaviporus(Berk & M.A. Curtis) Riebesehl & E. Langer'
