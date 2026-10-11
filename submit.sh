#!/bin/zsh
# usage: submit.sh <competition> <file> <message>
c=$1; f=$2; msg=$3; T=$(cat ~/.kaggle/access_token); H="Authorization: Bearer $T"
size=$(stat -f%z "$f"); mt=$(stat -f%m "$f")
r=$(curl -s -H "$H" -X POST -d "fileName=$(basename $f)" "https://www.kaggle.com/api/v1/competitions/$c/submissions/url/$size/$mt")
url=$(echo "$r" | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print(d.get("createUrl",""))' 2>/dev/null)
tok=$(echo "$r" | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print(d.get("token",""))' 2>/dev/null)
[ -z "$url" ] && { echo "URL step failed: $r" | head -c 400; echo; exit 1; }
curl -s -X PUT -T "$f" "$url" -o /dev/null -w "upload %{http_code}\n"
curl -s -H "$H" -X POST --data-urlencode "blobFileTokens=$tok" --data-urlencode "submissionDescription=$msg" "https://www.kaggle.com/api/v1/competitions/submissions/submit/$c" | head -c 400; echo
