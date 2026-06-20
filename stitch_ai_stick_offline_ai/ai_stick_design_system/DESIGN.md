---
name: Ai Stick Design System
colors:
  surface: '#fbf9f5'
  surface-dim: '#dbdad6'
  surface-bright: '#fbf9f5'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f5f3ef'
  surface-container: '#efeeea'
  surface-container-high: '#eae8e4'
  surface-container-highest: '#e4e2de'
  on-surface: '#1b1c1a'
  on-surface-variant: '#4d4635'
  inverse-surface: '#30312e'
  inverse-on-surface: '#f2f0ed'
  outline: '#7f7663'
  outline-variant: '#d0c5af'
  surface-tint: '#735c00'
  primary: '#735c00'
  on-primary: '#ffffff'
  primary-container: '#d4af37'
  on-primary-container: '#554300'
  inverse-primary: '#e9c349'
  secondary: '#77574d'
  on-secondary: '#ffffff'
  secondary-container: '#fed3c7'
  on-secondary-container: '#795950'
  tertiary: '#75584d'
  on-tertiary: '#ffffff'
  tertiary-container: '#cfab9f'
  on-tertiary-container: '#593f35'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#ffe088'
  primary-fixed-dim: '#e9c349'
  on-primary-fixed: '#241a00'
  on-primary-fixed-variant: '#574500'
  secondary-fixed: '#ffdbd0'
  secondary-fixed-dim: '#e7bdb1'
  on-secondary-fixed: '#2c160e'
  on-secondary-fixed-variant: '#5d4037'
  tertiary-fixed: '#ffdbce'
  tertiary-fixed-dim: '#e4beb2'
  on-tertiary-fixed: '#2b160f'
  on-tertiary-fixed-variant: '#5b4137'
  background: '#fbf9f5'
  on-background: '#1b1c1a'
  surface-variant: '#e4e2de'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 16px
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  gutter: 16px
  margin-mobile: 20px
---

## Brand & Style

The brand identity of the design system is anchored in a professional, high-utility aesthetic that balances luxury with functional precision. It is designed for users who require high-performance AI tools for translation and editing, demanding a workspace that feels both authoritative and approachable.

The design style follows a **Modern Corporate** approach with a **Tactile** edge. It utilizes the premium association of gold and deep earth tones while maintaining the strict clarity required for heavy-text interfaces. The visual language is defined by structured layouts, intentional whitespace, and a high-contrast palette that ensures accessibility across diverse multilingual contexts.

**Target Audience:** Professionals, content creators, and global travelers requiring reliable, high-fidelity AI processing and editing capabilities.

## Colors

The color palette is derived directly from the metallic gold and deep brown signature of the brand logo. 

- **Primary Gold (#D4AF37):** Reserved for high-impact actions and key brand indicators. It should be used sparingly for buttons, active states, and focus indicators.
- **Deep Brown (#5D4037):** Acts as the secondary functional color, providing strong contrast against the cream backgrounds. It is used for primary text and significant structural elements.
- **Cream Neutral (#FDFBF7):** The core background color, chosen to reduce eye strain compared to pure white, providing a "paper-like" quality that facilitates reading and editing.
- **Semantic Colors:** Use standard greens for success and reds for errors, but tint them slightly with brown/warm tones to ensure they harmonize with the overarching palette.

## Typography

This design system uses **Inter** for its exceptional legibility and extensive language support, which is critical for a translation-focused tool. 

- **Scale:** The hierarchy is tight and functional. We prioritize clear distinctions between headers and body copy to facilitate scanning.
- **Weight:** Use Bold (700) and Semi-bold (600) for hierarchy in headings. Regular (400) is used for all long-form reading and input text to ensure maximum clarity.
- **Editorial Intent:** For translation tasks, maintain a minimum body size of 16px to ensure readability for diverse user demographics.

## Layout & Spacing

The layout follows a **Fluid Grid** model optimized for mobile-first interaction. 

- **Mobile Rhythm:** Use a 20px outer margin to give content breathing room on small screens.
- **Vertical Rhythm:** A base-4 spacing system ensures consistent alignment. Standard vertical gaps between logical sections should be 24px (lg), while internal component spacing should be 8px (sm) or 12px.
- **Safe Areas:** Ensure all critical actions (like "Translate" or "Save") are placed within the thumb-zone and respect the bottom safe area on modern smartphones.

## Elevation & Depth

Visual hierarchy is managed through **Tonal Layers** and **Ambient Shadows**.

- **Surfaces:** Use `#FFFFFF` for cards and modals that need to sit above the `#FDFBF7` main background.
- **Shadows:** Use extremely soft, warm-tinted shadows. The shadow color should be a semi-transparent version of the secondary brown (`rgba(93, 64, 55, 0.08)`) rather than pure black.
- **Interaction:** Upon press, elements should visually "sink" slightly, achieved by reducing the shadow spread and slightly darkening the surface color.

## Shapes

The shape language is **Rounded**, reflecting a modern and user-friendly mobile experience.

- **Standard Elements:** Buttons, input fields, and small cards use a 0.5rem (8px) radius.
- **Large Elements:** Bottom sheets and large containers use 1rem (16px) or 1.5rem (24px) for a more organic, modern feel.
- **Selection Indicators:** Use pill-shaped (full-round) geometry for tags, chips, and toggles to differentiate them from primary action buttons.

## Components

### Buttons
- **Primary:** Solid Gold background (`#D4AF37`) with White text. Use for the main call-to-action (e.g., "Start Translation").
- **Secondary:** Deep Brown outline with Deep Brown text. Use for secondary actions (e.g., "Edit", "Copy").
- **Ghost:** No background, Deep Brown text. Used for "Cancel" or "Dismiss" actions.

### Input Fields
- Background should be white with a 1px border of `#E7E2D8`.
- On focus, the border transitions to Primary Gold with a subtle 2px outer glow.
- Labels sit above the field in `label-md` style.

### Cards
- Used for translation history or language selection. 
- Feature a white background and the "Ambient Shadow" defined in Elevation. 
- 16px internal padding.

### Chips
- Used for language selection (e.g., "English", "Spanish"). 
- Pill-shaped with a light beige background (`#F4F1EA`) and `body-sm` text.

### Toolbars
- Top navigation should be clean with minimal iconography. 
- Bottom toolbars for editing should use high-contrast icons (Deep Brown) against the cream background for maximum visibility.