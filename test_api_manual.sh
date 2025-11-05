#!/bin/bash

API_URL="http://localhost:8080"

echo "🧪 Début des tests automatiques de l'API..."

# Test 1 — Endpoint racine
echo "➡️  Test /"
response_root=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/)
if [ "$response_root" -eq 200 ]; then
    echo "✅ Test racine réussi"
else
    echo "❌ Test racine échoué (code $response_root)"
fi

# Test 2 — Endpoint /health
echo "➡️  Test /health"
response_health=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/health)
if [ "$response_health" -eq 200 ]; then
    echo "✅ Test health réussi"
else
    echo "❌ Test health échoué (code $response_health)"
fi

# Test 3 — Endpoint /predict
echo "➡️  Test /predict"
response_predict=$(curl -s -o /dev/null -w "%{http_code}" -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
        "surface": 75,
        "chambres": 3,
        "age_bien": 10,
        "quartier_score": 8,
        "distance_centre": 5.5
      }')

if [ "$response_predict" -eq 200 ]; then
    echo "✅ Test predict réussi"
else
    echo "❌ Test predict échoué (code $response_predict)"
fi

echo "------------------------------------"
echo "🧾 Résumé des tests :"
echo "  /          → $response_root"
echo "  /health    → $response_health"
echo "  /predict   → $response_predict"
echo "------------------------------------"

if [ "$response_root" -eq 200 ] && [ "$response_health" -eq 200 ] && [ "$response_predict" -eq 200 ]; then
    echo "🎉 Tous les tests sont PASSÉS avec succès !"
else
    echo "⚠️  Un ou plusieurs tests ont échoué."
fi