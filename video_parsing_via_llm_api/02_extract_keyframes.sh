for file in *.ts; do
    ffmpeg -y -i $file -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 2 "$file.jpg"
done
