#[derive(Debug, PartialEq)]
enum Term {
    Constant(String),
    Variable(String),
    Compound(String, Vec<Term>),
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

fn parse_term(input: &str) -> Option<Term> {
    let mut tokenizer = kohaku::Tokenizer::new(["(", ")", ",", "*", "?"]);
    let mut iter = tokenizer.tokenize(input).map_while(|x| x.ok());
    let term = take_term(&mut iter)?;
    iter.next().is_none().then_some(term)
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
    fn test_parse_term_17() {
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
    fn test_parse_term_18() {
        test_parse_term_fail("ab(c_d(e_f),g_h)))(");
    }
}
