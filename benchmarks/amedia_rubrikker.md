# Vurderingsrubrikker for Amedia-spørsmålene

To rubrikker for å vurdere kvaliteten på spørsmålene som er generert til
`results/amedia_fable_questions.json` (ett spørsmål per artikkel). Hvert
spørsmål vurderes **uavhengig** på to dimensjoner:

- **Rubrikk A – Spørsmålskvalitet:** Er spørsmålet et ekte høynivåspørsmål som
  tester dyp forståelse?
- **Rubrikk B – Forankring og unik kobling:** Må man referere til *nettopp
  denne* artikkelen for å svare, og er forholdet mellom artikkel og spørsmål
  en entydig 1-til-1-relasjon?

Begge rubrikkene bruker en skala fra **1 til 5**, der 5 er best.

---

## Rubrikk A – Spørsmålskvalitet

**Formål:** Måle om spørsmålet krever forståelse av sammenhenger, årsaker,
konsekvenser, motiver, hovedbudskap eller implikasjoner – i motsetning til
overfladisk faktagjenfinning med ett-ords svar.

| Nivå | Betegnelse | Kjennetegn |
|------|------------|------------|
| **5** | Utmerket | Krever tydelig forståelse av høyere orden (årsak, konsekvens, motiv, sammenheng, implikasjon eller hovedbudskap). Presist og entydig formulert på korrekt norsk. Har et klart, ikke-trivielt svar som forutsetter at leseren syntetiserer flere opplysninger fra teksten. |
| **4** | God | Krever forståelse utover ren faktagjenfinning, men er enten litt for bredt/smalt formulert, eller har mindre språklige svakheter. Svaret krever fortsatt tolkning. |
| **3** | Middels | Blandet: noe forståelse kreves, men svaret kan langt på vei hentes som ett enkelt faktum. Eventuelt er formuleringen noe uklar eller tvetydig. |
| **2** | Svak | I hovedsak et overfladisk faktaspørsmål med ett-ords eller svært kort svar, eller uklart/dårlig formulert. |
| **1** | Utilstrekkelig | Trivielt, meningsløst, uforståelig, grammatisk galt, eller ikke et reelt spørsmål. |

**Eksempler**
- *Nivå 5:* «Hvorfor mener lederen i Eidanger IL at synkehullene på
  fotballbanen ikke er grunn til stor bekymring, og hva er den antatte årsaken
  til at de oppstår?» – krever både årsaksforståelse og tolkning av lederens
  vurdering.
- *Nivå 1–2:* «Hvor dypt er synkehullet?» – ett-ords faktasvar, ingen
  forståelse kreves.

---

## Rubrikk B – Forankring og unik kobling (1-til-1)

**Formål:** Måle i hvilken grad man må referere eksplisitt til *denne*
artikkelen for å svare, og om spørsmålet er så spesifikt at nettopp denne
artikkelen er den entydige og eneste beste kilden (unik 1-til-1-kobling
mellom artikkel og spørsmål).

| Nivå | Betegnelse | Kjennetegn |
|------|------------|------------|
| **5** | Entydig forankret | Svaret finnes eksplisitt i, eller kan kun utledes fra, denne artikkelen. Spørsmålet inneholder spesifikke holdepunkter (navn, hendelser, tall, påstander) som gjør at kun denne artikkelen kan besvare det. Klar 1-til-1-kobling: ingen annen artikkel eller allmennkunnskap gir svaret. |
| **4** | Godt forankret | Klart forankret i artikkelen og krever den for et presist svar, men enkelte detaljer kunne teoretisk finnes i svært like kilder, eller spørsmålet er marginalt for bredt til å være helt entydig. |
| **3** | Delvis forankret | Svaret finnes i artikkelen, men kan helt eller delvis også besvares fra allmennkunnskap eller andre artikler. Koblingen er ikke entydig – flere kilder kunne passe. |
| **2** | Svakt forankret | Kan i stor grad besvares uten denne artikkelen, eller spørsmålet er så generelt at mange artikler kunne være kilde. Ingen tydelig unik kobling. |
| **1** | Ikke forankret | Svaret finnes ikke i artikkelen, er ren allmennkunnskap, eller kan besvares fra tittelen alene. Ingen 1-til-1-kobling. |

**Eksempler**
- *Nivå 5:* Et spørsmål som nevner navngitte personer, konkrete tall eller
  spesifikke hendelser fra artikkelen, slik at man må lese akkurat denne
  teksten for å svare korrekt.
- *Nivå 1–2:* «Hvorfor er fotball populært i Norge?» eller «Hva er
  hovedstaden i Norge?» – allmennkunnskap uten forankring i noen bestemt
  artikkel.

---

## Slik bruker du rubrikkene

- Gi **to separate skårer** per spørsmål: én for Rubrikk A og én for Rubrikk B.
  De vurderer ulike ting og skal ikke slås sammen til én tallverdi før begge er
  satt.
- **Anbefalt terskel for benchmark-kvalitet:** et spørsmål regnes som godkjent
  når det får **≥ 4 på begge** rubrikkene. Spørsmål med **1–2 på minst én**
  rubrikk bør forkastes eller omskrives; **3** er en gråsone som bør vurderes
  manuelt.
- For en LLM-dommer: be modellen begrunne skåren kort på norsk før den oppgir
  tallet, og be om utdata som gyldig JSON, f.eks.:

```json
{
  "kvalitet_score": 5,
  "kvalitet_begrunnelse": "...",
  "forankring_score": 5,
  "forankring_begrunnelse": "...",
  "godkjent": true
}
```

- For **forankring (Rubrikk B)** bør dommeren få både spørsmålet *og* hele
  artikkelteksten, slik at den faktisk kan sjekke om svaret finnes i teksten og
  om koblingen er unik. For **kvalitet (Rubrikk A)** er spørsmålet alene som
  regel tilstrekkelig, men artikkelen kan gis som kontekst.
