#!/usr/bin/env python
"""
A simple parser for the GFF3 format.

Test with transcripts.gff3 from
http://www.broadinstitute.org/annotation/gebo/help/gff3.html.

Format specification source:
http://www.sequenceontology.org/gff3.shtml

Version 1.0
"""
from __future__ import with_statement
from collections import namedtuple
import gzip
import urllib

__author__ = "Uli Koehler"
__license__ = "Apache License v2.0"

# Initialized GeneInfo named tuple. Note: namedtuple is immutable
gffInfoFields = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
GFFRecord = namedtuple("GFFRecord", gffInfoFields)


def parseGFFAttributes(attributeString):
    """Parse the GFF3 attribute column and return a dict"""  #
    if attributeString == ".": return {}
    ret = {}
    for attribute in attributeString.split(";"):
        key, value = attribute.split("=")
        ret[urllib.unquote(key)] = urllib.unquote(value)
    return ret


def parseGFF3(filename):
    """
    A minimalistic GFF3 format parser.
    Yields objects that contain info about a single GFF3 feature.

    Supports transparent gzip decompression.
    """
    # Parse with transparent decompression
    openFunc = gzip.open if filename.endswith(".gz") else open
    with openFunc(filename) as infile:
        for line in infile:
            if line.startswith("#"): continue
            parts = line.strip().split("\t")
            # If this fails, the file format is not standard-compatible
            assert len(parts) == len(gffInfoFields)
            # Normalize data
            normalizedInfo = {
                "seqid": None if parts[0] == "." else urllib.unquote(parts[0]),
                "source": None if parts[1] == "." else urllib.unquote(parts[1]),
                "type": None if parts[2] == "." else urllib.unquote(parts[2]),
                "start": None if parts[3] == "." else int(parts[3]),
                "end": None if parts[4] == "." else int(parts[4]),
                "score": None if parts[5] == "." else float(parts[5]),
                "strand": None if parts[6] == "." else urllib.unquote(parts[6]),
                "phase": None if parts[7] == "." else urllib.unquote(parts[7]),
                "attributes": parseGFFAttributes(parts[8])
            }
            # Alternatively, you can emit the dictionary here, if you need mutability:
            yield normalizedInfo
            # yield GFFRecord(**normalizedInfo)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="The GFF3 input file (.gz allowed)")
    parser.add_argument("--print-records", action="store_true", help="Print all GeneInfo objects, not only")
    parser.add_argument("--filter-type", help="Ignore records not having the given type")
    args = parser.parse_args()
    # Execute the parser
    recordCount = 0
    for record in parseGFF3(args.file):
        # Apply filter, if any
        if args.filter_type and record.type != args.filter_type:
            continue
        # Print record if specified by the user
        if args.print_records:
            print(record)
        # Access attributes like this: my_strand = record.strand
        recordCount += 1
    print("Total records: %d", recordCount)
