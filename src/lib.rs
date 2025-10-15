use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::slice::Iter;

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

fn occurs_check(
    nsv: u64,
    s: &str,
    nst: u64,
    t: &Term,
    r: &HashMap<(u64, &str), (u64, &Term)>,
) -> bool {
    match t {
        Term::Compound(_, args) => args.iter().all(|c| occurs_check(nsv, &s, nst, &c, &r)),
        Term::Variable(s1) if nsv == nst && s == s1 => false,
        Term::Variable(s1) if r.contains_key(&(nst, s1.as_str())) => {
            let &(ns1, t1) = r.get(&(nst, s1.as_str())).unwrap();
            occurs_check(nsv, s, ns1, t1, r)
        }
        _ => true,
    }
}

fn unify<'a>(
    ns1: u64,
    t1: &'a Term,
    ns2: u64,
    t2: &'a Term,
    mut r: HashMap<(u64, &'a str), (u64, &'a Term)>,
) -> Option<HashMap<(u64, &'a str), (u64, &'a Term)>> {
    match (t1, t2) {
        (Term::Compound(s1, args1), Term::Compound(s2, args2))
            if s1 == s2 && args1.len() == args2.len() =>
        {
            let mut iter = args1.iter().zip(args2.iter());
            iter.try_fold(r, |r, (c1, c2)| unify(ns1, c1, ns2, c2, r))
        }
        (Term::Constant(s1), Term::Constant(s2)) if s1 == s2 => Some(r),
        (Term::Variable(s1), Term::Variable(s2)) if ns1 == ns2 && s1 == s2 => Some(r),
        (Term::Variable(s), t) if r.contains_key(&(ns1, s.as_str())) => {
            let &(ns3, t3) = r.get(&(ns2, s.as_str())).unwrap();
            unify(ns3, t3, ns2, t, r)
        }
        (t, Term::Variable(s)) if r.contains_key(&(ns2, s.as_str())) => {
            let &(ns3, t3) = r.get(&(ns2, s.as_str())).unwrap();
            unify(ns3, t3, ns1, t, r)
        }
        (Term::Variable(s), t) if occurs_check(ns1, &s, ns2, &t, &r) => {
            r.insert((ns1, s), (ns2, t));
            Some(r)
        }
        (t, Term::Variable(s)) if occurs_check(ns2, &s, ns1, &t, &r) => {
            r.insert((ns2, s), (ns1, t));
            Some(r)
        }
        _ => None,
    }
}

#[derive(Clone)]
struct State<'a> {
    cost: u64,
    namespace: u64,
    table: HashMap<(u64, &'a str), (u64, &'a Term)>,
    goals: Vec<(u64, &'a Term)>,
    rules_iter: Iter<'a, Rule>,
}

impl Eq for State<'_> {}

impl PartialEq for State<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl PartialOrd for State<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for State<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

struct Infer<'a> {
    rules: &'a [Rule],
    stack: Vec<Reverse<State<'a>>>,
    pq: BinaryHeap<Reverse<State<'a>>>,
}

impl<'a> Infer<'a> {
    fn push_state(&mut self, state: State<'a>) {
        if let Some(Reverse(s)) = self.pq.peek()
            && *s == state
        {
            self.stack.push(Reverse(state))
        } else {
            self.pq.push(Reverse(state))
        }
    }

    fn pop_state(&mut self) -> Option<State<'a>> {
        self.stack.pop().or_else(|| self.pq.pop()).map(|x| x.0)
    }
}

impl<'a> Iterator for Infer<'a> {
    type Item = (u64, HashMap<(u64, &'a str), (u64, &'a Term)>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut state = self.pop_state()?;
            if state.goals.is_empty() {
                return Some((state.cost, state.table));
            }
            let Some(Rule::Rule(cost_rule, head, body)) = state.rules_iter.next() else {
                continue;
            };
            self.push_state(state.clone());
            state.cost = state.cost + cost_rule;
            state.namespace += 1;
            let (namespace_goal, goal) = state.goals.pop().unwrap();
            let Some(table) = unify(state.namespace, head, namespace_goal, goal, state.table)
            else {
                continue;
            };
            state.table = table;
            for body_term in body.into_iter().rev() {
                state.goals.push((state.namespace, body_term));
            }
            state.rules_iter = self.rules.iter();
            self.push_state(state);
        }
    }
}

fn infer<'a>(goals: &'a [Term], rules: &'a [Rule]) -> Infer<'a> {
    Infer {
        rules: rules,
        stack: vec![],
        pq: BinaryHeap::from([Reverse(State {
            cost: 0,
            namespace: 0,
            table: HashMap::new(),
            goals: goals.into_iter().rev().map(|x| (0, x)).collect(),
            rules_iter: rules.iter(),
        })]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_term_1() {
        assert_eq!(
            parse_term("ab(c_d(e_f*),g_h?)"),
            Some(Term::Compound(
                String::from("ab"),
                vec![
                    Term::Compound(
                        String::from("c_d"),
                        vec![Term::Constant(String::from("e_f"))],
                    ),
                    Term::Variable(String::from("g_h")),
                ],
            ))
        );
    }

    #[test]
    fn test_parse_term_2() {
        assert_eq!(parse_term("ab(c_d(*e_f),?g_h)))("), None);
    }

    #[test]
    fn test_parse_term_3() {
        assert_eq!(parse_term("a*("), None);
    }

    #[test]
    fn test_parse_term_4() {
        assert_eq!(parse_term("a,a*)"), None);
    }

    #[test]
    fn test_parse_term_5() {
        assert_eq!(parse_term("a(b*(c*)"), None);
    }

    #[test]
    fn test_parse_term_6() {
        assert_eq!(parse_term("a)"), None);
    }

    #[test]
    fn test_parse_term_7() {
        assert_eq!(parse_term("a(b**"), None);
    }

    #[test]
    fn test_parse_term_8() {
        assert_eq!(parse_term("a(*)"), None);
    }

    #[test]
    fn test_parse_term_9() {
        assert_eq!(parse_term("(a*)"), None);
    }

    #[test]
    fn test_parse_term_10() {
        assert_eq!(parse_term("a*a"), None);
    }

    #[test]
    fn test_parse_term_11() {
        assert_eq!(parse_term("a(a*a)"), None);
    }

    #[test]
    fn test_parse_term_12() {
        assert_eq!(
            parse_term("f(a*, b*, x?)"),
            Some(Term::Compound(
                String::from("f"),
                vec![
                    Term::Constant(String::from("a")),
                    Term::Constant(String::from("b")),
                    Term::Variable(String::from("x")),
                ],
            ))
        );
    }

    #[test]
    fn test_parse_term_13() {
        assert_eq!(parse_term("f(a, b, X)"), None);
    }

    #[test]
    fn test_parse_term_14() {
        assert_eq!(parse_term("f(*a, *b, ?x)"), None);
    }

    #[test]
    fn test_parse_term_15() {
        assert_eq!(parse_term("ab(c_d(e_f),g_h)))("), None);
    }

    #[test]
    fn test_parse_term_16() {
        assert_eq!(parse_term("ab(c_d(e_f*),g_h?)))("), None);
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
                0,
                &parse_term("f(a* ,b* ,x? )").unwrap(),
                1,
                &parse_term("f(y? ,b* ,c* )").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([
                ((0, "x"), (1, &parse_term("c*").unwrap())),
                ((1, "y"), (0, &parse_term("a*").unwrap()))
            ])
        )
    }

    #[test]
    fn test_unify_2() {
        assert_eq!(
            unify(
                1,
                &parse_term("f(x? ,y? )").unwrap(),
                1,
                &parse_term("f(a* ,b* )").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([
                ((1, "x"), (1, &parse_term("a*").unwrap())),
                ((1, "y"), (1, &parse_term("b*").unwrap()))
            ])
        )
    }

    #[test]
    fn test_unify_3() {
        assert_eq!(
            unify(
                0,
                &parse_term("x?").unwrap(),
                0,
                &parse_term("y?").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([((0, "x"), (0, &parse_term("y?").unwrap()))])
        )
    }

    #[test]
    fn test_unify_4() {
        assert_eq!(
            unify(
                0,
                &parse_term("f(a*,b*)").unwrap(),
                1,
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
                0,
                &parse_term("x?").unwrap(),
                0,
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
                1,
                &parse_term("f(f(x?),g(y?))").unwrap(),
                1,
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
                1,
                &parse_term("g(x?,y?,x?)").unwrap(),
                1,
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
                0,
                &parse_term("x?").unwrap(),
                0,
                &parse_term("x?").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::new()
        )
    }

    #[test]
    fn test_unify_9() {
        assert_eq!(
            unify(
                0,
                &parse_term("x?").unwrap(),
                1,
                &parse_term("f(x?)").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([((0, "x"), (1, &parse_term("f(x?)").unwrap()))])
        )
    }

    #[test]
    fn test_unify_10() {
        assert_eq!(
            unify(
                0,
                &parse_term("x?").unwrap(),
                1,
                &parse_term("x?").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([((0, "x"), (1, &parse_term("x?").unwrap()))])
        )
    }

    #[test]
    fn test_unify_11() {
        assert_eq!(
            unify(
                0,
                &parse_term("f(f(x?),g(y?))").unwrap(),
                1,
                &parse_term("f(y?,x?)").unwrap(),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([
                ((1, "x"), (0, &parse_term("g(y?)").unwrap())),
                ((1, "y"), (0, &parse_term("f(x?)").unwrap()))
            ])
        )
    }

    #[test]
    fn test_unify_12() {
        assert_eq!(
            unify(
                0,
                &parse_term("f(f(x?), x?)").unwrap(),
                1,
                &parse_term("f(x?,x?)").unwrap(),
                HashMap::new()
            ),
            None
        )
    }

    #[test]
    fn test_infer_1_1() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(bob*, pat*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(0, HashMap::new())]
        );
    }

    #[test]
    fn test_infer_1_2() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(liz*, pat*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            []
        );
    }

    #[test]
    fn test_infer_1_3() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(tom*, ben*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            []
        );
    }

    #[test]
    fn test_infer_1_4() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(x?, liz*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                0,
                HashMap::from([((0, "x"), (1, &parse_term("tom*").unwrap()))])
            )]
        );
    }

    #[test]
    fn test_infer_1_5() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(bob*, y?).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [
                (
                    0,
                    HashMap::from([((0, "y"), (1, &parse_term("ann*").unwrap()))])
                ),
                (
                    0,
                    HashMap::from([((0, "y"), (1, &parse_term("pat*").unwrap()))])
                )
            ]
        );
    }

    #[test]
    fn test_infer_1_6() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(p?, q?).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [
                (
                    0,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("pam*").unwrap())),
                        ((0, "q"), (1, &parse_term("bob*").unwrap()))
                    ])
                ),
                (
                    0,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("tom*").unwrap())),
                        ((0, "q"), (1, &parse_term("bob*").unwrap()))
                    ])
                ),
                (
                    0,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("tom*").unwrap())),
                        ((0, "q"), (1, &parse_term("liz*").unwrap()))
                    ])
                ),
                (
                    0,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("bob*").unwrap())),
                        ((0, "q"), (1, &parse_term("ann*").unwrap()))
                    ])
                ),
                (
                    0,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("bob*").unwrap())),
                        ((0, "q"), (1, &parse_term("pat*").unwrap()))
                    ])
                ),
                (
                    0,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("pat*").unwrap())),
                        ((0, "q"), (1, &parse_term("jim*").unwrap()))
                    ])
                )
            ]
        );
    }

    #[test]
    fn test_infer_1_7() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(y?, jim*), parent(x?, y?).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                0,
                HashMap::from([
                    ((0, "y"), (1, &parse_term("pat*").unwrap())),
                    ((0, "x"), (2, &parse_term("bob*").unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_1_8() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(tom*, x?), parent(x?, y?).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [
                (
                    0,
                    HashMap::from([
                        ((0, "x"), (1, &parse_term("bob*").unwrap())),
                        ((0, "y"), (2, &parse_term("ann*").unwrap()))
                    ])
                ),
                (
                    0,
                    HashMap::from([
                        ((0, "x"), (1, &parse_term("bob*").unwrap())),
                        ((0, "y"), (2, &parse_term("pat*").unwrap()))
                    ])
                )
            ]
        )
    }

    #[test]
    fn test_infer_1_9() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(x?, ann*), parent(x?, pat*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                0,
                HashMap::from([((0, "x"), (1, &parse_term("bob*").unwrap()))])
            )]
        )
    }

    #[test]
    fn test_infer_1_10() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(pam*, x?), parent(x?, y?), parent(y?, jim*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                0,
                HashMap::from([
                    ((0, "x"), (1, &parse_term("bob*").unwrap())),
                    ((0, "y"), (2, &parse_term("pat*").unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_2() {
        let rules = &parse_rules(
            r#"
                [0] big(bear*).
                [0] big(elephant*).
                [0] small(cat*).
                [0] brown(bear*).
                [0] black(cat*).
                [0] gray(elephant*).
                [0] dark(z?) :- black(z?).
                [0] dark(z?) :- brown(z?).
            "#,
        )
        .unwrap();
        let query = &parse_query("dark(x?), big(x?).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "z"), (0, &parse_term("x?").unwrap())),
                    ((0, "x"), (2, &parse_term("bear*").unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_3_1() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
                [0] female(pam*).
                [0] male(tom*).
                [0] male(bob*).
                [0] female(liz*).
                [0] female(ann*).
                [0] female(pat*).
                [0] male(jim*).
                [0] offspring(y?, x?) :- parent(x?, y?).
                [0] mother(x?, y?) :- parent(x?, y?), female(x?).
                [0] grandparent(x?, z?) :- parent(x?, y?), parent(y?, z?).
                [0] sister(x?, y?) :- parent(z?, x?), parent(z?, y?), female(x?), different(x?, y?).
                [0] predecessor(x?, z?) :- parent(x?, z?).
                [0] predecessor(x?, z?) :- parent(x?, y?), predecessor(y?, z?).
            "#,
        )
        .unwrap();
        let query = &parse_query("predecessor(tom*, pat*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "x"), (0, &parse_term("tom*").unwrap())),
                    ((1, "z"), (0, &parse_term("pat*").unwrap())),
                    ((1, "y"), (2, &parse_term("bob*").unwrap())),
                    ((3, "x"), (2, &parse_term("bob*").unwrap())),
                    ((3, "z"), (0, &parse_term("pat*").unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_3_2() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
                [0] female(pam*).
                [0] male(tom*).
                [0] male(bob*).
                [0] female(liz*).
                [0] female(ann*).
                [0] female(pat*).
                [0] male(jim*).
                [0] offspring(y?, x?) :- parent(x?, y?).
                [0] mother(x?, y?) :- parent(x?, y?), female(x?).
                [0] grandparent(x?, z?) :- parent(x?, y?), parent(y?, z?).
                [0] sister(x?, y?) :- parent(z?, x?), parent(z?, y?), female(x?), different(x?, y?).
                [0] predecessor(x?, z?) :- parent(x?, z?).
                [0] predecessor(x?, z?) :- parent(x?, y?), predecessor(y?, z?).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(pam*, bob*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(0, HashMap::from([]))]
        )
    }

    #[test]
    fn test_infer_3_3() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
                [0] female(pam*).
                [0] male(tom*).
                [0] male(bob*).
                [0] female(liz*).
                [0] female(ann*).
                [0] female(pat*).
                [0] male(jim*).
                [0] offspring(y?, x?) :- parent(x?, y?).
                [0] mother(x?, y?) :- parent(x?, y?), female(x?).
                [0] grandparent(x?, z?) :- parent(x?, y?), parent(y?, z?).
                [0] sister(x?, y?) :- parent(z?, x?), parent(z?, y?), female(x?), different(x?, y?).
                [0] predecessor(x?, z?) :- parent(x?, z?).
                [0] predecessor(x?, z?) :- parent(x?, y?), predecessor(y?, z?).
            "#,
        )
        .unwrap();
        let query = &parse_query("mother(pam*, bob*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "x"), (0, &parse_term("pam*").unwrap())),
                    ((1, "y"), (0, &parse_term("bob*").unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_3_4() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
                [0] female(pam*).
                [0] male(tom*).
                [0] male(bob*).
                [0] female(liz*).
                [0] female(ann*).
                [0] female(pat*).
                [0] male(jim*).
                [0] offspring(y?, x?) :- parent(x?, y?).
                [0] mother(x?, y?) :- parent(x?, y?), female(x?).
                [0] grandparent(x?, z?) :- parent(x?, y?), parent(y?, z?).
                [0] sister(x?, y?) :- parent(z?, x?), parent(z?, y?), female(x?), different(x?, y?).
                [0] predecessor(x?, z?) :- parent(x?, z?).
                [0] predecessor(x?, z?) :- parent(x?, y?), predecessor(y?, z?).
            "#,
        )
        .unwrap();
        let query = &parse_query("grandparent(pam*, ann*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "x"), (0, &parse_term("pam*").unwrap())),
                    ((1, "z"), (0, &parse_term("ann*").unwrap())),
                    ((1, "y"), (2, &parse_term("bob*").unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_3_5() {
        let rules = &parse_rules(
            r#"
                [0] parent(pam*, bob*).
                [0] parent(tom*, bob*).
                [0] parent(tom*, liz*).
                [0] parent(bob*, ann*).
                [0] parent(bob*, pat*).
                [0] parent(pat*, jim*).
                [0] female(pam*).
                [0] male(tom*).
                [0] male(bob*).
                [0] female(liz*).
                [0] female(ann*).
                [0] female(pat*).
                [0] male(jim*).
                [0] offspring(y?, x?) :- parent(x?, y?).
                [0] mother(x?, y?) :- parent(x?, y?), female(x?).
                [0] grandparent(x?, z?) :- parent(x?, y?), parent(y?, z?).
                [0] sister(x?, y?) :- parent(z?, x?), parent(z?, y?), female(x?), different(x?, y?).
                [0] predecessor(x?, z?) :- parent(x?, z?).
                [0] predecessor(x?, z?) :- parent(x?, y?), predecessor(y?, z?).
            "#,
        )
        .unwrap();
        let query = &parse_query("grandparent(bob*, jim*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "x"), (0, &parse_term("bob*").unwrap())),
                    ((1, "z"), (0, &parse_term("jim*").unwrap())),
                    ((1, "y"), (2, &parse_term("pat*").unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_4_1() {
        let rules = &parse_rules(
            r#"
                [6] parent(pam*, bob*).
                [5] parent(tom*, bob*).
                [4] parent(tom*, liz*).
                [3] parent(bob*, ann*).
                [2] parent(bob*, pat*).
                [1] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(bob*, pat*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(2, HashMap::new())]
        );
    }

    #[test]
    fn test_infer_4_2() {
        let rules = &parse_rules(
            r#"
                [6] parent(pam*, bob*).
                [5] parent(tom*, bob*).
                [4] parent(tom*, liz*).
                [3] parent(bob*, ann*).
                [2] parent(bob*, pat*).
                [1] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(liz*, pat*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            []
        );
    }

    #[test]
    fn test_infer_4_3() {
        let rules = &parse_rules(
            r#"
                [6] parent(pam*, bob*).
                [5] parent(tom*, bob*).
                [4] parent(tom*, liz*).
                [3] parent(bob*, ann*).
                [2] parent(bob*, pat*).
                [1] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(tom*, ben*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            []
        );
    }

    #[test]
    fn test_infer_4_4() {
        let rules = &parse_rules(
            r#"
                [6] parent(pam*, bob*).
                [5] parent(tom*, bob*).
                [4] parent(tom*, liz*).
                [3] parent(bob*, ann*).
                [2] parent(bob*, pat*).
                [1] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(x?, liz*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                4,
                HashMap::from([((0, "x"), (1, &parse_term("tom*").unwrap()))])
            )]
        );
    }

    #[test]
    fn test_infer_4_5() {
        let rules = &parse_rules(
            r#"
                [6] parent(pam*, bob*).
                [5] parent(tom*, bob*).
                [4] parent(tom*, liz*).
                [3] parent(bob*, ann*).
                [2] parent(bob*, pat*).
                [1] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(bob*, y?).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [
                (
                    2,
                    HashMap::from([((0, "y"), (1, &parse_term("pat*").unwrap()))])
                ),
                (
                    3,
                    HashMap::from([((0, "y"), (1, &parse_term("ann*").unwrap()))])
                )
            ]
        );
    }

    #[test]
    fn test_infer_4_6() {
        let rules = &parse_rules(
            r#"
                [6] parent(pam*, bob*).
                [5] parent(tom*, bob*).
                [4] parent(tom*, liz*).
                [3] parent(bob*, ann*).
                [2] parent(bob*, pat*).
                [1] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(p?, q?).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [
                (
                    1,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("pat*").unwrap())),
                        ((0, "q"), (1, &parse_term("jim*").unwrap()))
                    ])
                ),
                (
                    2,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("bob*").unwrap())),
                        ((0, "q"), (1, &parse_term("pat*").unwrap()))
                    ])
                ),
                (
                    3,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("bob*").unwrap())),
                        ((0, "q"), (1, &parse_term("ann*").unwrap()))
                    ])
                ),
                (
                    4,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("tom*").unwrap())),
                        ((0, "q"), (1, &parse_term("liz*").unwrap()))
                    ])
                ),
                (
                    5,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("tom*").unwrap())),
                        ((0, "q"), (1, &parse_term("bob*").unwrap()))
                    ])
                ),
                (
                    6,
                    HashMap::from([
                        ((0, "p"), (1, &parse_term("pam*").unwrap())),
                        ((0, "q"), (1, &parse_term("bob*").unwrap()))
                    ])
                )
            ]
        );
    }

    #[test]
    fn test_infer_4_7() {
        let rules = &parse_rules(
            r#"
                [6] parent(pam*, bob*).
                [5] parent(tom*, bob*).
                [4] parent(tom*, liz*).
                [3] parent(bob*, ann*).
                [2] parent(bob*, pat*).
                [1] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(y?, jim*), parent(x?, y?).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                1 + 2,
                HashMap::from([
                    ((0, "y"), (1, &parse_term("pat*").unwrap())),
                    ((0, "x"), (2, &parse_term("bob*").unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_4_8() {
        let rules = &parse_rules(
            r#"
                [6] parent(pam*, bob*).
                [5] parent(tom*, bob*).
                [4] parent(tom*, liz*).
                [3] parent(bob*, ann*).
                [2] parent(bob*, pat*).
                [1] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(tom*, x?), parent(x?, y?).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [
                (
                    5 + 2,
                    HashMap::from([
                        ((0, "x"), (1, &parse_term("bob*").unwrap())),
                        ((0, "y"), (2, &parse_term("pat*").unwrap()))
                    ])
                ),
                (
                    5 + 3,
                    HashMap::from([
                        ((0, "x"), (1, &parse_term("bob*").unwrap())),
                        ((0, "y"), (2, &parse_term("ann*").unwrap()))
                    ])
                )
            ]
        )
    }

    #[test]
    fn test_infer_4_9() {
        let rules = &parse_rules(
            r#"
                [6] parent(pam*, bob*).
                [5] parent(tom*, bob*).
                [4] parent(tom*, liz*).
                [3] parent(bob*, ann*).
                [2] parent(bob*, pat*).
                [1] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(x?, ann*), parent(x?, pat*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                3 + 2,
                HashMap::from([((0, "x"), (1, &parse_term("bob*").unwrap()))])
            )]
        )
    }

    #[test]
    fn test_infer_4_10() {
        let rules = &parse_rules(
            r#"
                [6] parent(pam*, bob*).
                [5] parent(tom*, bob*).
                [4] parent(tom*, liz*).
                [3] parent(bob*, ann*).
                [2] parent(bob*, pat*).
                [1] parent(pat*, jim*).
            "#,
        )
        .unwrap();
        let query = &parse_query("parent(pam*, x?), parent(x?, y?), parent(y?, jim*).").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                6 + 2 + 1,
                HashMap::from([
                    ((0, "x"), (1, &parse_term("bob*").unwrap())),
                    ((0, "y"), (2, &parse_term("pat*").unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_5_1() {
        let rules = &parse_rules(
            r#"
                [2] p* :- q*.
                [1] q*.
                [1] p* :- r*.
                [3] r*.
            "#,
        )
        .unwrap();
        let query = &parse_query("p*.").unwrap();
        assert_eq!(
            infer(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(3, HashMap::from([])), (4, HashMap::from([]))]
        )
    }
}
