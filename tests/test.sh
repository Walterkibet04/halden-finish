#!/bin/bash
mkdir -p /logs/verifier
chmod 700 /logs/verifier
chmod -R a+rX /app 2>/dev/null
chmod a+x /app/finish 2>/dev/null
cd /tests
python3 -m pytest /tests/test_outputs.py -v --ctrf /logs/verifier/ctrf.json
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
