"""Tests for the generated GraphQL interface."""
from __future__ import annotations

from dal_lab.graphql_facade import GraphQLFacade
from dal_lab.meter import Meter, count_rows


def test_graphql_nested_query_follows_foreign_keys(con, catalog):
    # Arrange
    fac = GraphQLFacade(con, catalog)
    # Act
    out = fac.execute('{ customer_by_key(key: 7) { c_name nation { n_name } '
                      'orders(limit: 2) { o_orderkey } } }')
    # Assert
    cust = out["data"]["customer_by_key"]
    assert cust["c_name"] == "Customer#000000007" and len(cust["orders"]) == 2
    assert "n_name" in cust["nation"]


def test_graphql_where_filters_and_limits(con, catalog):
    fac = GraphQLFacade(con, catalog)
    out = fac.execute('{ customer(where: {c_acctbal: {lt: 0}}, limit: 5) { c_acctbal } }')
    rows = out["data"]["customer"]
    assert len(rows) == 5 and all(r["c_acctbal"] < 0 for r in rows)


def test_graphql_plain_schema_has_no_aggregates(con, catalog):
    out = GraphQLFacade(con, catalog).execute('{ customer_aggregate(op: "count") { value } }')
    assert out["errors"] and out["data"] is None


def test_graphql_aggregate_counts_groups(con, catalog):
    fac = GraphQLFacade(con, catalog, with_aggregates=True)
    out = fac.execute('{ nation_aggregate(op: "count") { value } }')
    assert out["data"]["nation_aggregate"][0]["value"] == 25


def test_count_rows_counts_nested_graphql_objects():
    delivered = {"data": {"a": [{"x": 1, "b": [{"y": 1}, {"y": 2}]}, {"x": 2, "b": []}]}}
    assert count_rows(delivered) == 4
    assert count_rows({"preview": [1, 2]}) == 2 and Meter().totals()["calls"] == 0


def test_relation_names_disambiguate_repeated_parent():
    from dal_lab.graphql_facade import relation_names
    fks = {("sales", "sold_date_sk"): ("date_dim", "d_date_sk"),
           ("sales", "ship_date_sk"): ("date_dim", "d_date_sk"),
           ("sales", "item_sk"): ("item", "i_item_sk")}
    assert relation_names(fks, "sales", "item_sk", "item") == ("item", "sales")
    assert relation_names(fks, "sales", "ship_date_sk", "date_dim") == \
        ("ship_date", "sales_via_ship_date")
