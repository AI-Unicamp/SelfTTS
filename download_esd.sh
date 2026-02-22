pip install gdown
gdown --id 1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v
unzip Emotional\ Speech\ Dataset\ \(ESD\).zip
rm Emotional\ Speech\ Dataset\ \(ESD\).zip
mkdir files
mv Emotion\ Speech\ Dataset/* files/
rm -r Emotion\ Speech\ Dataset/
rm -r __MACOSX/
cd files
rm -r 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 
cd ../
mkdir transcriptions
mv files/*/*.txt transcriptions/
cat transcriptions/*.txt > ./transcriptions.txt
rm -r transcriptions
cd files
for spk in 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020; do for style in Angry Happy Neutral Sad Surprise; do mv "${spk}/$style" "${spk}_${style}"; done; done
rm -r 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020
cd ../
