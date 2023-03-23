

```bash
for i in *.HEIC; do convert "$i" -crop 1600x1600+1600+700 -resize 128x128 "mangled_$i.jpg"; done
```