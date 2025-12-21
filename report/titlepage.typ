/// Generate a title page
///
/// - `title`: The title of the document
/// - `subtitle`: The subtitle of the document
/// - `course`: The course name
/// - `name`: The author name and metadata
/// - `bottom-text`: The text at the bottom of the page
#let title-page(title: none, subtitle: none, course: none, name: none, bottom-text: none, logos: (none, none)) = page("a4")[
  #v(2.5%)
  #align(center)[
    #box(
      height: 10%,
      grid(
        align: center + horizon,
        columns: (auto, auto),
        column-gutter: 2%,
        box(height: 100%, logos.at(0, default: none)),
        box(height: 60%, logos.at(1, default: none)),
      ),
    )
  ]
  #align(center)[#text(title, size: 32pt, font: ("Bodoni", "Source Han Serif"))]
  #align(center)[#text(subtitle, size: 24pt, font: ("Bodoni", "Source Han Serif"))]
  #align(
    center,
    text(size: 12pt, font: ("Bodoni", "Source Han Serif"))[
      #course \
      #name
    ],
  )
  #place(
    bottom + center,
    [
      #align(
        center,
        text(size: 12pt, font: ("Bodoni", "Source Han Serif"))[
          #bottom-text
        ],
      )
    ],
  )
]
