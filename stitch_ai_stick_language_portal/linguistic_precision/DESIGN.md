---
name: Linguistic Precision
colors:
  surface: '#f7f9fb'
  surface-dim: '#d8dadc'
  surface-bright: '#f7f9fb'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f4f6'
  surface-container: '#eceef0'
  surface-container-high: '#e6e8ea'
  surface-container-highest: '#e0e3e5'
  on-surface: '#191c1e'
  on-surface-variant: '#47464f'
  inverse-surface: '#2d3133'
  inverse-on-surface: '#eff1f3'
  outline: '#787680'
  outline-variant: '#c8c5d0'
  surface-tint: '#5b598c'
  primary: '#070235'
  on-primary: '#ffffff'
  primary-container: '#1e1b4b'
  on-primary-container: '#8683ba'
  inverse-primary: '#c4c1fb'
  secondary: '#006a61'
  on-secondary: '#ffffff'
  secondary-container: '#86f2e4'
  on-secondary-container: '#006f66'
  tertiary: '#000c1d'
  on-tertiary: '#ffffff'
  tertiary-container: '#122336'
  on-tertiary-container: '#7a8aa2'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e3dfff'
  primary-fixed-dim: '#c4c1fb'
  on-primary-fixed: '#181445'
  on-primary-fixed-variant: '#444173'
  secondary-fixed: '#89f5e7'
  secondary-fixed-dim: '#6bd8cb'
  on-secondary-fixed: '#00201d'
  on-secondary-fixed-variant: '#005049'
  tertiary-fixed: '#d3e4fe'
  tertiary-fixed-dim: '#b7c8e1'
  on-tertiary-fixed: '#0b1c30'
  on-tertiary-fixed-variant: '#38485d'
  background: '#f7f9fb'
  on-background: '#191c1e'
  surface-variant: '#e0e3e5'
typography:
  h1:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  h2:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: '0'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: '0'
  label-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '600'
    lineHeight: '1'
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 8px
  xs: 4px
  sm: 12px
  md: 16px
  lg: 24px
  xl: 40px
  gutter: 16px
  margin: 20px
---

## Brand & Style

The design system is anchored in the **Corporate / Modern** aesthetic, tailored specifically for high-end productivity and cross-cultural communication. It balances the technical sophistication of an AI tool with the cultural warmth required for language translation between English and Runyoro/Rutooro.

The personality is intellectual, reliable, and precise. It targets bilingual professionals, students, and linguists who require a tool that feels like a durable utility rather than a toy. The interface utilizes generous whitespace, purposeful motion, and a structured hierarchy to ensure the user feels in control of complex linguistic data.

## Colors

This design system uses a sophisticated palette designed for long-form reading and clarity.

- **Primary (Deep Indigo):** Used for headers, navigation elements, and brand-critical touchpoints to establish authority and trust.
- **Secondary (Vibrant Teal):** Reserved for primary calls-to-action, active states, and successful AI processing indicators.
- **Tertiary (Soft Slate):** Applied to secondary text, icons, and borders to provide structure without visual noise.
- **Neutral (Slate Gray Tints):** The background utilizes a very light slate tint to reduce eye strain compared to pure white, maintaining a "document" feel.

## Typography

The design system relies exclusively on **Inter** to leverage its exceptional legibility and systematic weight distribution. 

- **Headlines:** Use tighter letter-spacing and heavier weights to create a strong vertical rhythm.
- **Body Text:** Optimized for bilingual comparison; line heights are generous (1.5x - 1.6x) to ensure that characters in both English and Runyoro/Rutooro are easily distinguishable.
- **Labels:** Small-caps or heavy weights are used for UI metadata, such as language indicators (ENG vs. RUN).

## Layout & Spacing

The design system employs a **Fluid Grid** model optimized for mobile devices. It utilizes an 8px base unit to ensure consistent scaling across components.

- **Margins:** A standard 20px horizontal margin keeps content safe from screen edges.
- **Vertical Rhythm:** Content blocks (like translation cards) are separated by 16px (md) spacing, while internal elements use 8px or 12px.
- **Alignment:** Left-aligned text is preferred for readability, with primary actions anchored to the bottom of the viewport for easy thumb access.

## Elevation & Depth

Visual hierarchy is achieved through **Ambient Shadows** and **Tonal Layers**. 

The design system avoids heavy drop shadows, opting for a single "elevation" style for cards and modals: a soft, multi-layered shadow with a 10% opacity Indigo tint to bridge the element with the background. Surfaces are layered by shifting the background from the Slate base (Level 0) to pure white (Level 1) for interactive elements. This creates a subtle sense of "lifting" the translation results above the workspace.

## Shapes

The design system uses a **Rounded** shape language to soften the professional aesthetic and make the tool feel more approachable. 

- **Standard Components:** Buttons, input fields, and small cards use an 8px (0.5rem) radius.
- **Large Containers:** Bottom sheets and full-width modals use 16px (1rem) for a more distinct structural separation.
- **Interactive States:** Indicators for active selection (like language toggles) follow the same 8px rule to maintain geometric consistency.

## Components

- **Buttons:** Primary buttons use the Vibrant Teal background with white text. Secondary buttons use a Slate outline with Indigo text.
- **Translation Cards:** Surfaces are white with an 8px radius and subtle ambient shadow. They feature a vertical divider to separate source and target languages.
- **Input Fields:** Clean, slate-bordered boxes that expand vertically. The active state is indicated by a 2px Indigo border.
- **Language Chips:** Small, rounded-pill elements used to toggle between English and Runyoro/Rutooro, utilizing a subtle Slate fill for inactive and Teal for active.
- **AI Processing Bar:** A slim, pulsing teal line at the top of the input area to indicate active "AI Stick" computation.
- **Lists:** Clean rows with 1px slate dividers, using Chevron-right icons in Slate to indicate drill-down actions for dictionary definitions.