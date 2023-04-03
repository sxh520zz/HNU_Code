for file in `ls "/home/shixiaohan-toda/Desktop/HNU_Code-main/TTS_Series/TTS_data/"`
do
	p="/home/shixiaohan-toda/Desktop/HNU_Code-main/TTS_Series/TTS_data/"${file}
	new="/home/shixiaohan-toda/Desktop/HNU_Code-main/TTS_Series/TTS_data_1/"${file}
	sox $p -b 16 -e signed-integer $new
 
done    
