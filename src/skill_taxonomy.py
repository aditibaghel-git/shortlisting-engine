"""
Skill taxonomy: canonical skill -> list of aliases/synonyms/related terms.
Used for:
  1. Keyword extraction from JD/resume text (alias -> canonical mapping)
  2. Fuzzy matching to catch typos/variants
  3. "Semantic implication" - e.g. Express implies Node.js backend experience

This is deliberately focused on web/full-stack dev skills since the sample JD
is a "Junior Full Stack Developer Intern" role. Extend this dict for other domains.
"""

SKILL_TAXONOMY = {
    "javascript": ["javascript", "js", "es6", "ecmascript", "vanilla js"],
    "typescript": ["typescript", "ts"],
    "python": ["python", "py"],
    "java": ["java", "j2ee", "core java"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "sass", "scss", "less", "tailwind", "tailwindcss", "bootstrap"],
    "react": ["react", "reactjs", "react.js", "react native", "jsx", "redux", "next.js", "nextjs"],
    "angular": ["angular", "angularjs", "angular.js"],
    "vue": ["vue", "vuejs", "vue.js"],
    "node.js": ["node", "nodejs", "node.js", "express", "express.js", "expressjs", "nestjs", "koa"],
    "rest_api": ["rest", "rest api", "restful", "restful api", "api development", "api design", "graphql"],
    "mongodb": ["mongodb", "mongo", "mongoose", "nosql"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "mssql", "relational database", "rdbms"],
    "database": ["database", "db", "dbms", "database design", "database management"],
    "git": ["git", "github", "gitlab", "version control", "bitbucket"],
    "docker": ["docker", "containerization", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "ci_cd": ["ci/cd", "ci cd", "continuous integration", "continuous deployment", "jenkins", "github actions"],
    "testing": ["testing", "unit testing", "jest", "mocha", "chai", "pytest", "junit", "test driven development", "tdd"],
    "agile": ["agile", "scrum", "kanban", "sprint"],
    "data_structures": ["data structures", "algorithms", "dsa", "data structures and algorithms"],
    "oop": ["oop", "object oriented programming", "object-oriented"],
    "linux": ["linux", "unix", "bash", "shell scripting", "command line"],
    "django": ["django", "flask", "fastapi"],
    "c_cpp": ["c++", "c", "cpp"],
    "machine_learning": ["machine learning", "ml", "deep learning", "ai", "artificial intelligence",
                          "tensorflow", "pytorch", "scikit-learn", "nlp"],
    "figma": ["figma", "ui/ux", "ux design", "ui design", "adobe xd", "wireframing"],
}

# Soft skills are tracked separately from technical skills. They are never counted
# toward the required-skill match ratio (a resume shouldn't be penalized in the
# core technical score for not literally saying "communication"), but they are
# still extracted and can be shown as a supplementary note.
SOFT_SKILL_TAXONOMY = {
    "communication": ["communication", "teamwork", "collaboration", "leadership", "presentation skills"],
    "problem_solving": ["problem solving", "problem-solving", "analytical skills", "critical thinking"],
}
SOFT_ALIAS_TO_CANONICAL = {}
for _canonical, _aliases in SOFT_SKILL_TAXONOMY.items():
    for _alias in _aliases:
        SOFT_ALIAS_TO_CANONICAL[_alias.lower()] = _canonical

# Reverse map: alias -> canonical skill (all lowercase)
ALIAS_TO_CANONICAL = {}
for canonical, aliases in SKILL_TAXONOMY.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical

# Skills that "imply" other skills for semantic scoring boosts
# e.g. knowing Express strongly implies Node.js backend competency
IMPLIES = {
    "node.js": ["rest_api"],
    "django": ["python", "rest_api"],
    "react": ["javascript"],
    "angular": ["typescript", "javascript"],
    "vue": ["javascript"],
    "mongodb": ["database"],
    "sql": ["database"],
}


def get_all_canonical_skills():
    return list(SKILL_TAXONOMY.keys())
