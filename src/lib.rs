use std::collections::HashMap;

#[derive(Debug, PartialEq)]
enum Term {
    Constant(String),
    Variable(String),
    Compound(String, Vec<Term>),
}

#[derive(Debug, PartialEq)]
enum Rule {
    Rule(u64, Term, Vec<Term>),
}

fn take_term_args<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Vec<Term>> {
    let term = take_term(iter)?;
    match iter.next()? {
        "," => {
            let mut args = take_term_args(iter)?;
            args.insert(0, term);
            Some(args)
        }
        ")" => Some(vec![term]),
        _ => None,
    }
}

fn take_term<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Term> {
    let label = iter.next()?;
    match iter.next()? {
        "*" => Some(Term::Constant(String::from(label))),
        "?" => Some(Term::Variable(String::from(label))),
        "(" => {
            let args = take_term_args(iter)?;
            Some(Term::Compound(String::from(label), args))
        }
        _ => None,
    }
}

fn take_query<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Vec<Term>> {
    let term = take_term(iter)?;
    match iter.next()? {
        "," => {
            let mut args = take_query(iter)?;
            args.insert(0, term);
            Some(args)
        }
        "." => Some(vec![term]),
        _ => None,
    }
}

fn take_body<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Vec<Term>> {
    let term = take_term(iter)?;
    match iter.next()? {
        "," => {
            let mut args = take_body(iter)?;
            args.insert(0, term);
            Some(args)
        }
        "." => Some(vec![term]),
        _ => None,
    }
}

fn take_rule<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Rule> {
    let cost = iter.next()?.parse().ok()?;
    let _ = (iter.next()? == "]").then_some(())?;
    let head = take_term(iter)?;
    match iter.next()? {
        ":-" => Some(Rule::Rule(cost, head, take_body(iter)?)),
        "." => Some(Rule::Rule(cost, head, Vec::new())),
        _ => None,
    }
}

fn take_rules<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Vec<Rule>> {
    match iter.next() {
        Some("[") => {
            let rule = take_rule(iter)?;
            let mut rules = take_rules(iter)?;
            rules.insert(0, rule);
            Some(rules)
        }
        None => Some(vec![]),
        _ => None,
    }
}

fn parse_term(input: &str) -> Option<Term> {
    let mut tokenizer = kohaku::Tokenizer::new(["(", ")", ",", "*", "?"]);
    let mut iter = tokenizer.tokenize(input).map_while(|x| x.ok());
    let term = take_term(&mut iter)?;
    iter.next().is_none().then_some(term)
}

fn parse_query(input: &str) -> Option<Vec<Term>> {
    let mut tokenizer = kohaku::Tokenizer::new(["(", ")", ",", "*", "?", "."]);
    let mut iter = tokenizer.tokenize(input).map_while(|x| x.ok());
    let query = take_query(&mut iter)?;
    iter.next().is_none().then_some(query)
}

fn parse_rules(input: &str) -> Option<Vec<Rule>> {
    let mut tokenizer = kohaku::Tokenizer::new(["(", ")", ",", "*", "?", ".", ":-", "[", "]"]);
    let mut iter = tokenizer.tokenize(input).map_while(|x| x.ok());
    let rules = take_rules(&mut iter)?;
    iter.next().is_none().then_some(rules)
}

fn occurs_check(s: &str, t: &Term, r: &HashMap<&str, &Term>) -> bool {
    match t {
        Term::Compound(_, args) => args.iter().all(|c| occurs_check(&s, &c, &r)),
        Term::Variable(s1) if s == s1 => false,
        Term::Variable(s1) if r.contains_key(s1.as_str()) => {
            occurs_check(s, r.get(s1.as_str()).unwrap(), r)
        }
        _ => true,
    }
}

fn unify<'a>(
    t1: &'a Term,
    t2: &'a Term,
    mut r: HashMap<&'a str, &'a Term>,
) -> Option<HashMap<&'a str, &'a Term>> {
    match (t1, t2) {
        (Term::Compound(s1, args1), Term::Compound(s2, args2))
            if s1 == s2 && args1.len() == args2.len() =>
        {
            let mut iter = args1.iter().zip(args2.iter());
            iter.try_fold(r, |r, (c1, c2)| unify(c1, c2, r))
        }
        (Term::Constant(s1), Term::Constant(s2)) if s1 == s2 => Some(r),
        (Term::Variable(s1), Term::Variable(s2)) if s1 == s2 => Some(r),
        (Term::Variable(s), t) | (t, Term::Variable(s)) if r.contains_key(s.as_str()) => {
            unify(r.get(s.as_str()).unwrap(), t, r)
        }
        (Term::Variable(s), t) | (t, Term::Variable(s)) if occurs_check(&s, &t, &r) => {
            r.insert(&s, t);
            Some(r)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parse_term_success(input: &str, output: Term) {
        let term = parse_term(input);
        assert_eq!(term, Some(output));
    }

    fn test_parse_term_fail(input: &str) {
        let term = parse_term(input);
        assert_eq!(term, None);
    }

    #[test]
    fn test_parse_term_1() {
        test_parse_term_success(
            "ab(c_d(e_f*),g_h?)",
            Term::Compound(
                String::from("ab"),
                vec![
                    Term::Compound(
                        String::from("c_d"),
                        vec![Term::Constant(String::from("e_f"))],
                    ),
                    Term::Variable(String::from("g_h")),
                ],
            ),
        );
    }

    #[test]
    fn test_parse_term_2() {
        test_parse_term_fail("ab(c_d(*e_f),?g_h)))(");
    }

    #[test]
    fn test_parse_term_3() {
        test_parse_term_fail("a*(");
    }

    #[test]
    fn test_parse_term_4() {
        test_parse_term_fail("a,a*)");
    }

    #[test]
    fn test_parse_term_5() {
        test_parse_term_fail("a(b*(c*)");
    }

    #[test]
    fn test_parse_term_6() {
        test_parse_term_fail("a)");
    }

    #[test]
    fn test_parse_term_7() {
        test_parse_term_fail("a(b**");
    }

    #[test]
    fn test_parse_term_8() {
        test_parse_term_fail("a(*)");
    }

    #[test]
    fn test_parse_term_9() {
        test_parse_term_fail("(a*)");
    }

    #[test]
    fn test_parse_term_10() {
        test_parse_term_fail("a*a");
    }

    #[test]
    fn test_parse_term_11() {
        test_parse_term_fail("a(a*a)");
    }

    #[test]
    fn test_parse_term_12() {
        test_parse_term_success(
            "f(a*, b*, x?)",
            Term::Compound(
                String::from("f"),
                vec![
                    Term::Constant(String::from("a")),
                    Term::Constant(String::from("b")),
                    Term::Variable(String::from("x")),
                ],
            ),
        );
    }

    #[test]
    fn test_parse_term_13() {
        test_parse_term_fail("f(a, b, X)");
    }

    #[test]
    fn test_parse_term_14() {
        test_parse_term_fail("f(*a, *b, ?x)");
    }

    #[test]
    fn test_parse_term_15() {
        test_parse_term_fail("ab(c_d(e_f),g_h)))(");
    }

    #[test]
    fn test_parse_term_16() {
        test_parse_term_fail("ab(c_d(e_f*),g_h?)))(");
    }

    #[test]
    fn test_parse_query_1() {
        let query = parse_query("f(a*, b*, x?).");
        assert_eq!(
            query,
            Some(vec![Term::Compound(
                String::from("f"),
                vec![
                    Term::Constant(String::from("a")),
                    Term::Constant(String::from("b")),
                    Term::Variable(String::from("x")),
                ]
            )])
        );
    }

    #[test]
    fn test_parse_query_2() {
        let query = parse_query("f(a*, b*, x?), g(c*, y?), h(d*).");
        assert_eq!(
            query,
            Some(vec![
                Term::Compound(
                    String::from("f"),
                    vec![
                        Term::Constant(String::from("a")),
                        Term::Constant(String::from("b")),
                        Term::Variable(String::from("x")),
                    ]
                ),
                Term::Compound(
                    String::from("g"),
                    vec![
                        Term::Constant(String::from("c")),
                        Term::Variable(String::from("y")),
                    ]
                ),
                Term::Compound(String::from("h"), vec![Term::Constant(String::from("d"))])
            ])
        );
    }

    #[test]
    fn test_parse_rules_1() {
        let rules = parse_rules("[2]a* :- b*, c?.   \n[4]d*.\n");
        assert_eq!(
            rules,
            Some(vec![
                Rule::Rule(
                    2,
                    Term::Constant(String::from("a")),
                    vec![
                        Term::Constant(String::from("b")),
                        Term::Variable(String::from("c"))
                    ]
                ),
                Rule::Rule(4, Term::Constant(String::from("d")), vec![])
            ])
        );
    }

    #[test]
    fn test_take_rules_2() {
        let rules = parse_rules("[2]a:-b,C.\n");
        assert_eq!(rules, None);
    }

    #[test]
    fn test_take_rules_3() {
        let rules = parse_rules("[2]a:-b,C.  \n[4]d.\n");
        assert_eq!(rules, None);
    }

    #[test]
    fn test_take_rules_4() {
        let rules = parse_rules("[2]*a :- *b, ?c.   \n[4]*d.\n");
        assert_eq!(rules, None);
    }

    #[test]
    fn test_unify_1() {
        assert_eq!(
            unify(
                &parse_term("f(a* ,b* ,x? )").unwrap(),
                &parse_term("f(y? ,b* ,c* )").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([
                ("x", &parse_term("c*").unwrap()),
                ("y", &parse_term("a*").unwrap())
            ])
        )
    }

    #[test]
    fn test_unify_2() {
        assert_eq!(
            unify(
                &parse_term("f(x? ,y? )").unwrap(),
                &parse_term("f(a* ,b* )").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([
                ("x", &parse_term("a*").unwrap()),
                ("y", &parse_term("b*").unwrap())
            ])
        )
    }

    #[test]
    fn test_unify_3() {
        assert_eq!(
            unify(
                &parse_term("x?").unwrap(),
                &parse_term("y?").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([("x", &parse_term("y?").unwrap())])
        )
    }

    #[test]
    fn test_unify_4() {
        assert_eq!(
            unify(
                &parse_term("f(a*,b*)").unwrap(),
                &parse_term("f(x?,x?)").unwrap(),
                HashMap::new()
            ),
            None
        )
    }

    #[test]
    fn test_unify_5() {
        assert_eq!(
            unify(
                &parse_term("x?").unwrap(),
                &parse_term("f(x?)").unwrap(),
                HashMap::new()
            ),
            None
        )
    }

    #[test]
    fn test_unify_6() {
        assert_eq!(
            unify(
                &parse_term("f(f(x?),g(y?))").unwrap(),
                &parse_term("f(y?,x?)").unwrap(),
                HashMap::new()
            ),
            None
        )
    }

    #[test]
    fn test_unify_7() {
        assert_eq!(
            unify(
                &parse_term("g(x?,y?,x?)").unwrap(),
                &parse_term("g(f(x?),f(y?),y?)").unwrap(),
                HashMap::new()
            ),
            None
        )
    }

    #[test]
    fn test_unify_8() {
        assert_eq!(
            unify(
                &parse_term("x?").unwrap(),
                &parse_term("x?").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::new()
        )
    }
}
