"""Update all numbers in the report generator script to match the new pipeline results."""
from pathlib import Path

p = Path("scripts/generate_report_doc.py")
content = p.read_text(encoding="utf-8")

replacements = [
    # Raw totals
    ('"3,485"',          '"3,782"'),
    ('3,485 raw pairs',  '3,782 raw pairs'),
    ('3,057 removed',    '2,901 removed'),
    ('3,057',            '2,901'),
    ('out of 3,485 raw pairs, 3,057 were removed and 428 were kept.',
     'out of 3,782 raw pairs, 2,901 were removed and 881 were kept.'),
    # Valid pairs
    ('"428"',            '"881"'),
    ('428 valid',        '881 valid'),
    ('428 clean pairs',  '881 clean pairs'),
    ('428 entries, 1,727 lines', '881 entries, 3,562 lines'),
    ('"428  (187 modified)"',  '"881  (174 modified)"'),
    # Cleaning description
    ('The 428 pairs that passed filtering were all passed through a cleaning pipeline.  '
     '187 of them (43.7%) had something fixed.  The other 241 were already clean.',
     'The 881 pairs that passed filtering were all passed through a cleaning pipeline.  '
     '174 of them (19.7%) had something fixed.  The other 707 were already clean.'),
    # Augmentation numbers
    ('428 original + 316 augmented', '881 original + 323 augmented'),
    ('"428 original + 316 new augmented pairs',
     '"881 original + 323 new augmented pairs'),
    ('Original 428 pairs  +  316 new augmented pairs  =  744 total pairs.',
     'Original 881 pairs  +  323 new augmented pairs  =  1,204 total pairs.'),
    ('"Total new pairs from augmentation:  316"',
     '"Total new pairs from augmentation:  323"'),
    ('"+316"',           '"+323"'),
    # Total corpus
    ('"744"',            '"1,204"'),
    ('428 + 316 = 744',  '881 + 323 = 1,204'),
    ('744 total pairs.', '1,204 total pairs.'),
    ('"744 pairs',       '"1,204 pairs'),
    ('744 pairs (original + augmented)', '1,204 pairs (original + augmented)'),
    # Splits
    ('"632"',            '"1,023"'),
    ('632 training pairs',  '1,023 training pairs'),
    ('632 / 74 / 38',    '1,023 / 120 / 61'),
    ('train=632, val=74, test=38', 'train=1,023, val=120, test=61'),
    ('"74 + 38"',        '"120 + 61"'),
    ('"74"',             '"120"'),
    ('"38"',             '"61"'),
    ('74 validation pairs', '120 validation pairs'),
    ('38 test pairs',    '61 test pairs'),
    ('632 train', '1,023 train'),
    ('74 pairs (10%)', '120 pairs (10%)'),
    ('38 pairs ( 5%)', '61 pairs ( 5%)'),
    # Section description: 428 clean pairs is small
    ('428 clean pairs is a small dataset',
     '881 clean pairs is a reasonable dataset'),
    # Stage table
    ('"After removing bad pairs",        "428"',
     '"After removing bad pairs",        "881"'),
    ('"After cleaning",                  "428"',
     '"After cleaning",                  "881"'),
    ('"Augmented pairs created",         "+316"',
     '"Augmented pairs created",         "+323"'),
    ('"Total corpus",                    "744"',
     '"Total corpus",                    "1,204"'),
    ('"Training set (85%)",              "632"',
     '"Training set (85%)",              "1,023"'),
    ('"Validation set (10%) + Test (5%)","74 + 38"',
     '"Validation set (10%) + Test (5%)","120 + 61"'),
    # Files table
    ('"428 clean pairs"',    '"881 clean pairs"'),
    ('"632 training pairs"', '"1,023 training pairs"'),
    ('"74 validation pairs"','"120 validation pairs"'),
    ('"38 test pairs"',      '"61 test pairs"'),
    # Glossary
    ('37 terminology entries', '425 terminology entries'),
    ('37 terms)',          '425 terms)'),
    ('"37 entries"',       '"425 entries"'),
    ('37-term glossary',   '425-term glossary'),
    # Named entities
    ('55 named entities',  '58 named entities'),
    ('55 entries',         '58 entries'),
    # Pipeline diagram
    ('3,485\", "",',       '3,782\", "",'),
    ('428 valid pairs remain', '881 valid pairs remain'),
    ('187 pairs fixed, 428 total', '174 pairs fixed, 881 total'),
    ('428 + 316 = 744 total pairs', '881 + 323 = 1,204 total pairs'),
    ('632 train / 74 val / 38 test', '1,023 train / 120 val / 61 test'),
    # Final summary box
    ('3,485", False', '3,782", False'),
    ('3,057", False', '2,901", False'),
    ('428", False',   '881", False'),
    ('+316", False',  '+323", False'),
    ('744", True',    '1,204", True'),
    ('632 pairs (85%)', '1,023 pairs (85%)'),
    # Next steps
    ('the 38 held-out test pairs', 'the 61 held-out test pairs'),
    ('The 744 pairs are fed', 'The 1,204 pairs are fed'),
    # Rejection description
    ('3,057 removed (too short, symbols, bad ratio)',
     '2,901 removed (wrong-language examples, symbols, bad ratio)'),
    # Cleaning stats section
    ('"Total raw pairs extracted:  3,485"',
     '"Total raw pairs extracted:  3,782"'),
]

changed = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        changed += 1
        print(f"  OK: {old[:60]!r}")

p.write_text(content, encoding="utf-8")
print(f"\nDone — {changed} replacements applied.")
