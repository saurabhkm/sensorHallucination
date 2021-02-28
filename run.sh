valRatio=0.5
panLR=0.0003
msLR=0.0003
tsLR=0.00015
hallLR=0.0003
finalLR=0.00015
batchSize=64
instance=1
mod1=PAN
mod2=MS
name=$mod1$mod2
trainTestBS=30

T=5
alpha=0.5
Lambda=0.5

python singleStream.py -valRatio $valRatio -learningRate $panLR -batchSize $batchSize -nEpochs 100 -instance $instance -modality $mod1 -trainTestBS $trainTestBS -name $name #-plot
python singleStream.py -valRatio $valRatio -learningRate $msLR -batchSize $batchSize -nEpochs 100 -instance $instance -modality $mod2 -trainTestBS $trainTestBS -name $name #-plot
python twoStream.py -valRatio $valRatio -learningRate $tsLR -batchSize $batchSize -nEpochs 200 -instance $instance -modality1 $mod1 -modality2 $mod2 -trainTestBS $trainTestBS -name $name #-plot
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
python hallucinate.py -valRatio $valRatio -learningRate $hallLR -batchSize $batchSize -T $T -Lambda $Lambda -alpha $alpha -nEpochs 150 -instance $instance -modality1 $mod1 -modality2 $mod2 -trainTestBS $trainTestBS -name $name #-plot
python finalNet.py -valRatio $valRatio -learningRate $finalLR -batchSize $batchSize -nEpochs 200 -instance $instance -modality1 $mod1 -modality2 $mod2 -trainTestBS $trainTestBS -name $name #-plot