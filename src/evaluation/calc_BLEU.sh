set -eux

LANG=$1 # ja
DIRECTION=$2 #en-ja

HOME_PATH=/Users/abe-k/restricted_translation/wat2022_restricted/
GOLD_PATH=${HOME_PATH}data/ASPEC-JE/test_gold.${LANG}
# normal
# FILTERED_OUTPUTS_PATH=${HOME_PATH}work/filtered_outputs/${DIRECTION}
# remove backslash
# FILTERED_OUTPUTS_PATH=${HOME_PATH}work/filtered_outputs_includeslash/${DIRECTION}
# no filtered
FILTERED_OUTPUTS_PATH=${HOME_PATH}submitted_outputs/TMU/${DIRECTION}

for OUTPUT_PATH in $( ls ${FILTERED_OUTPUTS_PATH})
do
echo ${OUTPUT_PATH}
        
if [ $DIRECTION = 'en-ja' ]; then
    # en-ja
    cat ${FILTERED_OUTPUTS_PATH}/${OUTPUT_PATH} | sacrebleu ${GOLD_PATH} -l ${DIRECTION} -tok ja-mecab
else
    # ja-en
    cat ${FILTERED_OUTPUTS_PATH}/${OUTPUT_PATH} | sacrebleu ${GOLD_PATH} -l ${DIRECTION}  
fi

done
