"""Confirmed reporting labels; immutable acquisition keys and training roles stay intact."""
from campaign_common import probability_cohorts

GROUPS = {'Clear': ['june24_clear', 'june25_clear'], 'LM': ['june24_lowmotor'],
          'HM': ['june24_highmotor'], 'C+B+M': ['june25_highmotor'],
          'T': ['june26_plain_dirty'], 'TB': ['june26_bubbles'], 'TBM': ['june26_bubbles_highmotor']}

def display_condition(condition):
    return next(name for name, keys in GROUPS.items() if condition in keys)

def display_cohorts(rows):
    return {name: probability_cohorts([r for r in rows if r['condition'] in keys])['overall']
            for name, keys in GROUPS.items() if any(r['condition'] in keys for r in rows)}
