## Первая лабораторная работа


- Создан сайт, который работает по ссылке https://nikitakondrat.github.io/
- Использован mkdocs и в качестве theme выбран simple-blog
- Также работает поиск и справа вверху есть ссылка еа профиль GitHub


Так выглядят настройки mkdocs.yml на 21.02.2026


    site_name: Портфолио Никиты Кондрата
    site_url: https://nikitakondrat.github.io/
    site_description: "Студент ИТМО"
    site_author: Никита Кондрат

    nav:
      - Main page: index.md
      - About author: about.md
      - Labworks 2 semester:
           labwork1 Tutorial: labwork1.md
           labwork2 Tutorial: labwork2.md

    theme:
      name: simple-blog
      avatar: images/avatar.jpg
      author: Никита Кондрат

    repo_url: https://github.com/NikitaKondrat/
    repo_name: NikitaKondrat/

    plugins:
      - search:
          lang: ["ru"]
          separator: '[\s\-]+'