---
title: PCI Express - Wikipedia
id: pci-express-wikipedia
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-allreduce-ceiling-and-definition
created: '2026-10-27T13:20:42.775041Z'
updated: '2026-10-27T19:00:31.682860Z'
source: https://en.wikipedia.org/wiki/PCI_Express
source_domain: en.wikipedia.org
fetched_at: '2026-10-27T13:20:42.773634Z'
fetch_provider: builtin
status: review
type: note
tier: institutional
content_type: article
deprecated: false
summary: From Wikipedia, the free encyclopedia
utility_score: 12.0
---

PCI Express - Wikipedia
Jump to content
From Wikipedia, the free encyclopedia
Computer expansion bus standard
Not to be confused with
PCI-X
or
UCIe
.
For Engineering, Procurement, Construction and Installation, see
EPCI
.
PCI Express
Peripheral Component Interconnect Express
Year created
2003
;
23
years ago
(
2003
)
Created by
Intel
Dell
HP
IBM
Supersedes
PCI
PCI-X
AGP
Width in bits
1 per lane, up to 16 lanes
[
1
]
No.
of devices
1 on each endpoint of each connection
[
a
]
Speed
Dual simplex
, up to 242
GB/s
Style
Serial
Hotplugging interface
Optional (supported with
ExpressCard
,
OCuLink
,
CFexpress
or
U.2
)
External interface
Optional (supported with
OCuLink
or other forms of
PCI Express External Cabling
, tunneled over
USB4
and
Thunderbolt
)
Website
pcisig
.com
A PCIe 3.0 x8
host bus adapter
Various slots on a
computer motherboard
, from top to bottom:
PCI Express x4
PCI Express x16
PCI Express x1
PCI Express x16
Conventional PCI
(32-bit, 5
V)
PCI Express
(
Peripheral Component Interconnect Express
), officially abbreviated as
PCIe
,
[
2
]
is a high-speed standard used to connect hardware components inside computers. It is designed to replace older
expansion bus
standards such as
PCI
,
PCI-X
and
AGP
. Developed and maintained by the
PCI-SIG
(PCI Special Interest Group), PCIe is commonly used to connect
graphics cards
,
sound cards
,
Wi-Fi
and
Ethernet
adapters, and storage devices such as
solid-state drives
and
hard disk drives
.
[
3
]
Compared to earlier standards, PCIe supports faster data transfer, uses fewer pins, takes up less space, and allows devices to be added or removed while the computer is running (
hot swapping
). It also includes better error detection and supports newer features like
I/O virtualization
for advanced computing needs.
[
4
]
PCIe connections are made through
lanes
, which are pairs of conductors that send and receive data. Devices can use one or more lanes depending on how much data they need to transfer.
[
5
]
PCIe technology is also used in laptop expansion cards (like
ExpressCard
) and in storage connectors such as
M.2
,
U.2
, and
SATA Express
.
Architecture
[
edit
]
Example of the PCI
Express topology:
white "junction boxes" represent PCI
Express device downstream ports. The gray ones represent upstream ports.
[
6
]
:
7
PCI Express x1 card containing a PCI Express switch (covered by a small
heat sink
), which creates multiple endpoints out of one endpoint and lets multiple devices share it
The PCIe slots on a motherboard are often labeled with the number of PCIe lanes they have. Sometimes what may seem like a large slot may only have a few lanes. For instance, a x16 slot with only 4 PCIe lanes (bottom slot) is quite common.
[
7
]
Conceptually, the PCI Express bus is a high-speed
serial
replacement of the older PCI/PCI-X bus.
[
8
]
One of the key differences between the PCI Express bus and the older PCI is the bus topology; PCI uses a shared
parallel
bus
architecture, in which the PCI host and all devices share a common set of address, data, and control lines. In contrast, PCI Express is based on point-to-point
topology
, with separate
serial
links connecting every device to the
root complex
(host). Because of its shared bus topology, access to the older PCI bus is arbitrated (in the case of multiple masters), and limited to one master at a time, in a single direction. Furthermore, the older PCI clocking scheme limits the bus clock to the slowest peripheral on the bus (regardless of the devices involved in the bus transaction). In contrast, a PCI Express bus link supports
full-duplex
communication between any two endpoints, with no inherent limitation on concurrent access across multiple endpoints.
In terms of bus protocol, PCI Express communication is encapsulated in packets. The work of packetizing and de-packetizing data and status-message traffic is handled by the transaction layer of the PCI Express port (described later). Radical differences in electrical signaling and bus protocol require the use of a different mechanical form factor and expansion connectors (and thus, new motherboards and new adapter boards); PCI slots and PCI Express slots are not interchangeable. At the software level, PCI Express preserves
backward compatibility
with PCI; legacy PCI system software can detect and configure newer PCI Express devices without explicit support for the PCI Express standard, though new PCI Express features are inaccessible.
The PCI Express link between two devices can vary in size from one to 16
lanes
. In a multi-lane link, the packet data is striped across lanes, and peak data throughput scales with the overall link width. The lane count is automatically negotiated during device initialization and can be restricted by either endpoint. For example, a single-lane PCI Express (x1) card can be inserted into a multi-lane slot (x4, x8, etc.), and the initialization cycle auto-negotiates the highest mutually supported lane count. The link can dynamically down-configure itself to use fewer lanes, providing a failure tolerance in case bad or unreliable lanes are present. The PCI Express standard defines link widths of x1, x2, x4, x8, and x16. Up to and including PCIe 5.0, x12, and x32 links were defined as well but virtually
[
clarification needed
]
never used.
[
9
]
This allows the PCI Express bus to serve both cost-sensitive applications where high throughput is not needed, and performance-critical applications such as 3D graphics, networking (
10 Gigabit Ethernet
or multiport
Gigabit Ethernet
), and enterprise storage (
SAS
or
Fibre Channel
). Slots and connectors are only defined for a subset of these widths, with link widths in between using the next larger physical slot size.
As a point of reference, a PCI-X (133
MHz 64-bit) device and a PCI Express
1.0 device using four lanes (x4) have roughly the same peak single-direction transfer rate of 1064
MB/s. The PCI Express bus has the potential to perform better than the PCI-X bus in cases where multiple devices are transferring data simultaneously, or if communication with the PCI Express peripheral is
bidirectional
.
Interconnect
[
edit
]
A PCI Express link between two devices consists of one or more lanes, which are
dual simplex
channels using two
differential signaling
pairs.
[
6
]
:
3
PCI Express devices communicate via a logical connection called an
interconnect
[
10
]
or
link
. A link is a point-to-point communication channel between two PCI Express ports allowing both of them to send and receive ordinary PCI requests (configuration, I/O or memory read/write) and
interrupts
(
INTx
,
MSI or MSI-X
). At the physical level, a link is composed of one or more
lanes
.
[
10
]
Low-speed peripherals (such as an
802.11
Wi-Fi
card
) use a single-lane (x1) link, while a graphics adapter typically uses a much wider and therefore faster 16-lane (x16) link.
Lane
[
edit
]
A lane is composed of two
differential signaling
pairs, with one pair for receiving data and the other for transmitting. Thus, each lane is composed of four wires or
signal traces
. Conceptually, each lane is used as a
full-duplex
byte stream
, transporting data packets in eight-bit "byte" format simultaneously in both directions between endpoints of a link.
[
11
]
Physical PCI Express links may contain 1, 4, 8 or 16 lanes.
[
12
]
[
6
]
:
4,
5
[
10
]
Lane counts are written with an "x" prefix (for example, "x8" represents an eight-lane card or slot), with x16 being the largest size in common use.
[
13
]
Lane sizes are also referred to via the terms "width" or "by" e.g., an eight-lane slot could be referred to as a "by 8" or as "8 lanes wide."
For mechanical card sizes, see
below
.
Serial bus
[
edit
]
This section
does not
cite
any
sources
.
Please help
improve this section
by
adding citations to reliable sources
. Unsourced material may be challenged and
removed
.
(
March 2018
)
(
Learn how and when to remove this message
)
The bonded serial
bus
architecture was chosen over the traditional parallel bus because of the inherent limitations of the latter, including
half-duplex
operation, excess signal count, and inherently lower
bandwidth
due to
timing skew
. Timing skew results from separate electrical signals within a parallel interface traveling through conductors of different lengths, on potentially different
printed circuit board
(PCB) layers, and at possibly different
signal velocities
. Despite being transmitted simultaneously as a single
word
, signals on a parallel interface have different travel duration and arrive at their destinations at different times. When the interface clock period is shorter than the largest time difference between signal arrivals, recovery of the transmitted word is no longer possible. Since timing skew over a parallel bus can amount to a few nanoseconds, the resulting bandwidth limitation is in the range of hundreds of megahertz.
Highly simplified
topologies of the Legacy PCI Shared (Parallel) Interface and the PCIe Serial Point-to-Point Interface
[
14
]
A serial interface does not exhibit timing skew because there is only one differential signal in each direction within each lane, and there is no external clock signal since clocking information is
embedded
within the serial signal itself. As such, typical bandwidth limitations on serial signals are in the multi-gigahertz range. PCI Express is one example of the general trend toward replacing parallel buses with serial interconnects; other examples include
Serial ATA
(SATA),
USB
,
Serial Attached SCSI
(SAS),
FireWire
(IEEE 1394), and
RapidIO
. In digital video, examples in common use are
DVI
,
HDMI
, and
DisplayPort
, but they were replacements for analog VGA, not for a parallel bus.
Multichannel serial design increases flexibility with its ability to allocate fewer lanes for slower devices.
Switch
[
edit
]
A
PCIe switch
is like a
network switch
, except that it works with PCIe connections. In the tree topology is one
upstream
port where it connects to a higher-level device (another switch or directly a PCIe bus) and a number of
downstream
ports. The switch is responsible for receiving a packet from one port and forwarding it to an appropriate, different port. By doing packet-switching, the bandwidth is effectively shared across multiple downstream ports (compared to static lane bifurcation). A more advanced multi-host topology contains multiple upstream ports.
[
15
]
Form factors
[
edit
]
PCI Express add-in card
[
edit
]
Intel P3608 NVMe flash SSD, PCIe add-in card
A PCI Express add-in card fits into a slot of its physical size or larger (with x16 as the largest used), but may not fit into a smaller PCI Express slot; for example, a x16 card may not fit into a x4 or x8 slot. Some slots use open-ended sockets to permit physically longer cards and negotiate the best available electrical and logical connection.
The number of lanes actually connected to a slot may also be fewer than the number supported by the physical slot size. An example is a x16 slot that runs at x4, which accepts any x1, x2, x4, x8 or x16 card, but provides only four lanes. Its specification may read as "x16 (x4 mode)" or "x16 (x4 signal)", while "mechanical @ electrical" notation (e.g. "x16
@
x4") is also common.
[
citation needed
]
The advantage is that such slots can accommodate a larger range of PCI Express cards without requiring motherboard hardware to support the full transfer rate. Standard mechanical sizes are x1, x4, x8, and x16. Cards using a number of lanes other than the standard mechanical sizes need to physically fit the next larger mechanical size (e.g. an x2 card uses the x4 size, or an x12 card uses the x16 size).
The cards themselves are designed and manufactured in various sizes. For example,
solid-state drives
(SSDs) that come in the form of PCI Express cards often use
HHHL
(half height, half length) and
FHHL
(full height, half length) to describe the physical dimensions of the card. The concept of "full" and "half" heights and lengths are inherited from Conventional PCI.
[
16
]
[
17
]
PCI express add-in card dimensions
[
18
]
PCI card type
Dimensions
Notes
(mm)
(in)
Height
Standard (Full)
111.15
4.376
Fits 3U chassis.
Low profile (Half)
68.90
2.731
Fits 2U chassis.
Length
Full
312.00
4.376 × 12.283
Enough space for x16.
Three-quarter
254.00
4.376 × 10.00
Enough space for x16.
Half
167.65
4.376 × 6.60
Enough space for x16.
Width
Single slot
18.71
0.737
Fits 1U chassis if rotated.
Dual slot
39.04
1.537
Fits 1U chassis if rotated.
Triple slot
59.36
2.337
Fits 2U chassis if rotated.
The length levels beside full are not a PCIe standard, but only a manufacturer agreement. Half length provides sufficient space for a x16 connector. Below that narrower data connectors need to be used.
These dimensions can be freely mixed and matched, but larger dimensions tend to co-occur.
There is a fixed distance of
57.15 millimetres (2.250
in)
between the connector's key notch (middle ridge diving data and power) and the end of the card, which may be covered by an end plate with a screw-hole for installing onto the computer case. This fixed length ensures that cards do not protrude out of the chassis.
The slot spacing is exactly
0.8 inches (20
mm)
on
ATX
motherboards.
For further specifications of the slot, see
#Physical layer
below.
Non-standard video card form factors
[
edit
]
Modern (since
c.
2012
[
19
]
) gaming
video cards
usually exceed the height as well as thickness specified in the PCI Express standard, due to the need for more capable and quieter
cooling fans
, as gaming video cards often emit hundreds of watts of heat.
[
20
]
Modern computer cases are often wider to accommodate these taller cards, but not always. Since full-length cards (312
mm) are uncommon, modern cases sometimes cannot accommodate them. The thickness of these cards also typically occupies the space of 2 to 5
[
21
]
PCIe slots. In fact, even the methodology of how to measure the cards varies between vendors, with some including the metal bracket size in dimensions and others not.
For instance, comparing three high-end video cards released in 2020: a
Sapphire
Radeon RX 5700 XT
card measures 135
mm in height (excluding the metal bracket), which exceeds the PCIe standard height by 28
mm,
[
22
]
another Radeon RX 5700 XT card by
XFX
measures 55
mm thick (i.e. 2.7 PCI slots at 20.32
mm), taking up 3 PCIe slots,
[
23
]
while an
Asus
GeForce RTX 3080
video card takes up two slots and measures 140.1
mm × 318.5
mm × 57.8
mm, exceeding PCI Express's maximum height, length, and thickness respectively.
[
24
]
Pinout
[
edit
]
The following table identifies the conductors on each side of the
edge connector
on a PCI Express card. The solder side of the
printed circuit board
(PCB) is the A-side, and the component side is the B-side.
[
25
]
PRSNT1# and PRSNT2# pins must be slightly shorter than the rest, to ensure that a hot-plugged card is fully inserted. The WAKE# pin uses full voltage to wake the computer, but must be
pulled high
from the standby power to indicate that the card is wake capable.
[
26
]
PCI Express connector pinout (x1, x4, x8 and x16 variants)
Pin
Side B
Side A
Description
Pin
Side B
Side A
Description
0
1
+12
V
PRSNT1#
Must connect to farthest PRSNT2# pin
50
HSOp(8)
Reserved
Lane 8 transmit data,
+ and −
0
2
+12
V
+12
V
Main power pins
51
HSOn(8)
Ground
0
3
+12
V
+12
V
52
Ground
HSIp(8)
Lane 8 receive data,
+ and −
0
4
Ground
Ground
53
Ground
HSIn(8)
0
5
SMCLK
TCK
SMBus
and
JTAG
port pins
54
HSOp(9)
Ground
Lane 9 transmit data,
+ and −
0
6
SMDAT
TDI
55
HSOn(9)
Ground
0
7
Ground
TDO
56
Ground
HSIp(9)
Lane 9 receive data,
+ and −
0
8
+3.3
V
TMS
57
Ground
HSIn(9)
0
9
TRST#
+3.3
V
58
HSOp(10)
Ground
Lane 10 transmit data,
+ and −
10
+3.3
V aux
+3.3
V
Aux power &
standby power
59
HSOn(10)
Ground
11
WAKE#
PERST#
Link reactivation; fundamental reset
[
27
]
60
Ground
HSIp(10)
Lane 10 receive data,
+ and −
Key notch
61
Ground
HSIn(10)
12
CLKREQ#
Ground
Clock request signal
[
28
]
62
HSOp(11)
Ground
Lane 11 transmit data,
+ and −
13
Ground
REFCLK+
Reference clock differential pair
63
HSOn(11)
Ground
14
HSOp(0)
REFCLK−
Lane 0 transmit data,
+ and −
64
Ground
HSIp(11)
Lane 11 receive data,
+ and −
15
HSOn(0)
Ground
65
Ground
HSIn(11)
16
Ground
HSIp(0)
Lane 0 receive data,
+ and −
66
HSOp(12)
Ground
Lane 12 transmit data,
+ and −
17
PRSNT2#
HSIn(0)
67
HSOn(12)
Ground
18
Ground
Ground
68
Ground
HSIp(12)
Lane 12 receive data,
+ and −
PCI Express x1 cards end at pin 18
69
Ground
HSIn(12)
19
HSOp(1)
Reserved
Lane 1 transmit data,
+ and −
70
HSOp(13)
Ground
Lane 13 transmit data,
+ and −
20
HSOn(1)
Ground
71
HSOn(13)
Ground
21
Ground
HSIp(1)
Lane 1 receive data,
+ and −
72
Ground
HSIp(13)
Lane 13 receive data,
+ and −
22
Ground
HSIn(1)
73
Ground
HSIn(13)
23
HSOp(2)
Ground
Lane 2 transmit data,
+ and −
74
HSOp(14)
Ground
Lane 14 transmit data,
+ and −
24
HSOn(2)
Ground
75
HSOn(14)
Ground
25
Ground
HSIp(2)
Lane 2 receive data,
+ and −
76
Ground
HSIp(14)
Lane 14 receive data,
+ and −
26
Ground
HSIn(2)
77
Ground
HSIn(14)
27
HSOp(3)
Ground
Lane 3 transmit data,
+ and −
78
HSOp(15)
Ground
Lane 15 transmit data,
+ and −
28
HSOn(3)
Ground
79
HSOn(15)
Ground
29
Ground
HSIp(3)
Lane 3 receive data,
+ and −
"Power brake", active-low to reduce device power
80
Ground
HSIp(15)
Lane 15 receive data,
+ and −
30
PWRBRK#
HSIn(3)
81
PRSNT2#
HSIn(15)
31
PRSNT2#
Ground
82
Reserved
Ground
32
Ground
Reserved
PCI Express x4 cards end at pin 32
33
HSOp(4)
Reserved
Lane 4 transmit data,
+ and −
34
HSOn(4)
Ground
35
Ground
HSIp(4)
Lane 4 receive data,
+ and −
36
Ground
HSIn(4)
37
HSOp(5)
Ground
Lane 5 transmit data,
+ and −
38
HSOn(5)
Ground
39
Ground
HSIp(5)
Lane 5 receive data,
+ and −
40
Ground
HSIn(5)
41
HSOp(6)
Ground
Lane 6 transmit data,
+ and −
42
HSOn(6)
Ground
43
Ground
HSIp(6)
Lane 6 receive data,
+ and −
Legend
44
Ground
HSIn(6)
Power
Supplies power to the card
45
HSOp(7)
Ground
Lane 7 transmit data,
+ and −
Sense
Tied together on card
46
HSOn(7)
Ground
Ground
Zero volt reference
47
Ground
HSIp(7)
Lane 7 receive data,
+ and −
Open drain
May be pulled low or sensed by multiple cards
[
29
]
48
PRSNT2#
HSIn(7)
Host-to-card
Signal from the motherboard to the card
49
Ground
Ground
Card-to-host
Signal from the card to the motherboard
PCI Express x8 cards end at pin 49
Reserved
Not presently used
Power
[
edit
]
The main
12
V
power supply for the PCIe slot is pins B2, B3 (side B) and pins A2, A3 (side A). Power standby
3.3
V
is pin B10 and A10. PCIe x1 cards can draw up to
25
W
and x16 graphics cards can draw up to
75
W
, combined.
[
30
]
Slot power
[
edit
]
All PCI express cards may consume up to
3
A
at
+3.3
V
(
9.9
W
). The amount of +12
V and total power they may consume depends on the form factor and the role of the card:
[
31
]
:
35–36
[
32
]
[
33
]
x1 cards are limited to 0.5
A at +12
V (6
W) and 10
W combined.
x4 and wider cards are limited to 2.1
A at +12
V (25
W) and 25
W combined.
A full-sized x1 card may draw up to the 25
W limits after initialization and software configuration as a high-power device.
A full-sized x16 graphics card may draw up to 5.5
A at +12
V (66
W) and 75
W combined after initialization and software configuration as a high-power device.
[
26
]
:
38–39
6- and 8-pin power connectors
[
edit
]
8-pin (left) and 6-pin (right)
power connectors
used on PCI Express cards
Optional connectors add
75
W
(6-pin) or
150
W
(8-pin) of +12
V power for up to
300
W
total (
2 @ 75
W + 1 @ 150
W
).
Sense0 pin is connected to ground by the cable or power supply, or float on board if cable is not connected.
Sense1 pin is connected to ground by the cable or power supply, or float on board if cable is not connected.
Some cards use two 8-pin connectors, allowing 375
W total (
1 @ 75
W + 2 @ 150
W
). This was newly standardized in PCI Express 4.0 CEM of 2018, though it was already in use before then.
[
18
]
The 8-pin PCI Express connector should not be confused with the
EPS12V
connector, which is mainly used for powering SMP and multi-core systems. The power connectors are variants of the Molex Mini-Fit Jr. series connectors.
[
34
]
Molex Mini-Fit Jr. part numbers
[
34
]
Pins
Receptacle
on PS cable
Right-angle
header on PCB
6-pin
45559-0002
45558-0003
8-pin
45587-0004
45586-0005, 45586-0006
6-pin power connector (75
W)
[
35
]
Pin
Description
0
1
+12
V
0
2
Not connected (usually +12
V as well)
0
3
+12
V
—
N/a
0
4
Ground
0
5
Sense
0
6
Ground
—
N/a
8-pin power connector (150
W)
[
36
]
[
37
]
[
38
]
Pin
Description
0
1
+12
V
0
2
+12
V
0
3
+12
V
0
4
Sense 1 (8-pin connected
[
A
]
)
0
5
Ground
0
6
Sense 0 (6-pin or 8-pin connected)
0
7
Ground
0
8
Ground
6 pin power connector pin map
8 pin power connector pin map
↑
When a 6-pin connector is plugged into an 8-pin receptacle the card is notified by a missing
Sense1
that it may only use up to 75
W.
12VHPWR connector
[
edit
]
This section is an excerpt from
12VHPWR
.
[
edit
]
16-pin 12VHPWR connector
12VHPWR and 12v-2x6 connector pinout
16-pin 12V-2×6 socket on a
Palit
GeForce
RTX 5090
Gamerock
The
16-pin 12VHPWR connector
is a standard for connecting
graphics processing units
(GPUs) to
computer power supplies
for up to 600
W power delivery. It was introduced by
Nvidia
in 2022 to supersede the previous
6- and 8-pin power connectors
for GPUs. The stated aim was to cater to the increasing power requirements of Nvidia GPUs. The connector was formally adopted as part of PCI Express 5.
[
39
]
The connector was replaced by a minor revision called 12V-2x6, introduced in 2023 with PCIe CEM 5.1 and PCIe ECN 6.0,
[
40
]
[
41
]
which changed the GPU- and PSU-side sockets to ensure that the sense pins only make contact if the power pins are seated properly. The cables and their plugs remained unchanged.
[
42
]
The change is intended to prevent melting due to partial contact, but melting continued to be reported for GPUs with this new socket.
[
43
]
There is a significant change in power negotiation with a new sense pin added.
[
44
]
12VHPWR connectors are marked with an "H+" symbol, whereas 12V-2x6 connectors are marked with an "H++" symbol.
[
45
]
48VHPWR connector
[
edit
]
In 2021, PCIe Card Electromechanical (CEM, pronounced like "chem" in "chemistry") Specification introduced a connector for 48 Volts with two current-carrying contacts and four sense pins. It was retained into PCIe-CEM 5.1 of 2023.
[
46
]
The contacts are rated for 15 Amps continuous current. The 48VHPWR connector can carry 720 watts.
48VHPWR pinout, Plug side
P1
P2
+48V (15
A)
Ground (15
A)
S1
S2
S3
S4
CARD_PWR_STABLE
CARD_CBL_PRES#
SENSE0
SENSE1
Later
[
when?
]
it was removed and an incompatible 48V 1×2 connector was introduced where Sense0 and Sense1 are located farthest from each other.
Power excursion
[
edit
]
Power excursion refers to short peaks of power draw exceeding the rated maximum (sustained) power level. Since an add-on Engineering Change Notice (ECN) to PCIe-CEM 5.0, the additional power connectors need to be able to handle 100-microsecond power draw at 3× of maximum sustained power, reducing to 1× at the 1-second window level following a logarithmic line. Since PCIe-ECM 5.1, slot power has a similar excursion expansion at 2.5× over 100
μs. In CEM 5.1, the added excursion limit is only provided after software configuration, specifically the Set_​Slot_​Power_​Limit message. The ECN is part of ATX 3.0 and PCIe CEM 5.1 is part of ATX 3.1.
[
47
]
PCI Express Mini Card
[
edit
]
A
WLAN
PCI Express Mini Card and its connector
MiniPCI and MiniPCI Express cards in comparison
PCI Express Mini Card
(also known as
Mini PCI Express
,
Mini PCIe
,
Mini PCI-E
,
mPCIe
, and
PEM
), based on PCI Express, is a replacement for the
Mini PCI
form factor. It is developed by the
PCI-SIG
. The host device supports both PCI Express and
USB
2.0 connectivity, and each card may use either standard. Most laptop computers built between 2005 and 2013 use Mini PCI Express for expansion cards; however,
as of 2015
[
update
]
, many vendors have moved toward using the newer
M.2
form factor for this purpose.
[
48
]
Due to different dimensions, PCI Express Mini Cards are not physically compatible with standard full-size PCI Express slots; however, passive adapters exist that let them be used in full-size slots.
[
49
]
Physical dimensions
[
edit
]
Dimensions of PCI Express Mini Cards are 30
mm × 50.95
mm (width
× length) for a Full Mini Card. There is a 52-pin
edge connector
, consisting of two staggered rows on a 0.8
mm pitch. Each row has eight contacts, a gap equivalent to four contacts, then a further 18 contacts.
Boards
have a thickness of 1.0
mm, excluding the components. A "Half Mini Card" (sometimes abbreviated as HMC) is also specified, having approximately half the physical length of 26.8
mm. There are also half size mini PCIe cards that are 30 × 31.90
mm which is about half the length of a full size mini PCIe card.
[
50
]
[
51
]
Electrical interface
[
edit
]
PCI Express Mini Card edge connectors provide multiple connections and buses:
PCI Express x1 (with SMBus)
USB
2.0
Wires to diagnostics LEDs for wireless network (i.e.,
Wi-Fi
) status on computer's chassis
SIM
card for
GSM
and
WCDMA
applications (UIM signals on spec.)
Future extension for another PCIe lane
1.5
V and 3.3
V power
Mini-SATA (mSATA) variant
[
edit
]
A
Mushkin
Atlas 240GB
mSATA
Solid-state drive
Despite sharing the Mini PCI Express form factor, an
mSATA
slot is not necessarily electrically compatible with Mini PCI Express. For this reason, only certain notebooks are compatible with mSATA drives. Most compatible systems are based on Intel's Sandy Bridge processor architecture, using the Huron River platform. Notebooks such as Lenovo's ThinkPad T, W and X series, released in March–April 2011, have support for an mSATA SSD card in their
WWAN
card slot. The ThinkPad Edge E220s/E420s, and the Lenovo IdeaPad Y460/Y560/Y570/Y580 also support mSATA.
[
52
]
On the contrary, the L-series among others can only support M.2 cards using the PCIe standard in the WWAN slot.
Some notebooks (notably the
Asus Eee PC
, the
Apple
MacBook Air
, and the Dell mini9 and mini10) use a variant of the PCI Express Mini Card as an
SSD
. This variant uses the reserved and several non-reserved pins to implement SATA and IDE interface passthrough, keeping only USB, ground lines, and sometimes the core PCIe x1 bus intact.
[
53
]
This makes the "miniPCIe" flash and solid-state drives sold for netbooks largely incompatible with true PCI Express Mini implementations.
Also, the typical Asus miniPCIe SSD is 71
mm long, causing the Dell 51
mm model to often be (incorrectly) referred to as half length. A true 51
mm Mini PCIe SSD was announced in 2009, with two stacked PCB layers that allow for higher storage capacity. The announced design preserves the PCIe interface, making it compatible with the standard mini PCIe slot. No working product has yet been developed.
Intel has numerous desktop boards with the PCIe x1 Mini-Card slot that typically do not support mSATA SSD. A list of desktop boards that natively support mSATA in the PCIe x1 Mini-Card slot (typically multiplexed with a SATA port) is provided on the Intel Support site.
[
54
]
PCI Express M.2
[
edit
]
Main article:
M.2
M.2 replaces the mSATA standard and Mini PCIe.
[
55
]
Computer bus interfaces provided through the M.2 connector are PCI Express 3.0 or higher (up to four lanes), Serial ATA 3.0, and USB 3.0 (a single logical port for each of the latter two). It is up to the manufacturer of the M.2 host or device to choose which interfaces to support, depending on the desired level of host support and device type.
PCI Express External Cabling
[
edit
]
PCI Express External Cabling
(also known as
External PCI Express
,
Cabled PCI Express
, or
ePCIe
) specifications were released by the
PCI-SIG
in February 2007.
[
56
]
[
57
]
Standard cables and connectors have been defined for x1, x4, x8, and x16 link widths, with a transfer rate of 250
MB/s per lane. The PCI-SIG also expects the norm to evolve to reach 500
MB/s, as in PCI Express 2.0. An example of the uses of Cabled PCI Express is a metal enclosure, containing a number of PCIe slots and PCIe-to-ePCIe adapter circuitry. This device would not be possible had it not been for the ePCIe specification.
PCI Express OCuLink
[
edit
]
OCuLink
(standing for "optical-copper link", as
Cu
is the
chemical symbol
for
copper
) is an extension for the "cable version of PCI Express". Version 1.0 of OCuLink, released in Oct 2015, supports up to 4 PCIe 3.0 lanes (3.9
GB/s) over copper cabling; a
fiber optic
version may appear in the future.
The most recent version of OCuLink, OCuLink-2, supports 8
GB/s or 16
GB/s (PCIe 4.0 x4 or x8)
[
58
]
while the maximum bandwidth of a USB (
USB4 Gen 4
) or
Thunderbolt 5
connection is 10
GB/s.
OCulink is principally intended for PCIe (or SATA breakout) interconnections in servers, but also finds limited adoption on laptops for the connection of external GPU boxes.
[
59
]
CopprLink
[
edit
]
CopprLink uses SFF-TA-1032 connectors to support external PCIe cable for PCIe 5.0 x16 and PCIe 6.0 x16.
[
60
]
[
61
]
[
62
]
Derivative forms
[
edit
]
Numerous other form factors use, or are able to use, PCIe. These include:
Low-height card
ExpressCard
: Successor to the
PC Card
form factor (with x1 PCIe and USB 2.0; hot-pluggable)
PCI Express ExpressModule: A hot-pluggable modular form factor defined for servers and workstations
XQD card
: A PCI Express-based flash card standard by the
CompactFlash Association
with x2 PCIe
CFexpress
card: A PCI Express-based flash card by the CompactFlash Association in three form factors supporting 1 to 4 PCIe lanes
SD card: The
SD Express
bus, introduced in version 7.0 of the SD specification uses a x1 PCIe link
XMC
: Similar to the
CMC
/
PMC
form factor (VITA 42.3)
AdvancedTCA
: A complement to
CompactPCI
for larger applications; supports serial based backplane topologies
AMC
: A complement to the
AdvancedTCA
specification; supports processor and I/O modules on ATCA boards (x1, x2, x4 or x8 PCIe).
FeaturePak
: A tiny expansion card format (43
mm × 65
mm) for embedded and small-form-factor applications, which implements two x1 PCIe links on a high-density connector along with USB, I2C, and up to 100 points of I/O
Universal IO
: A variant from
Super Micro Computer
Inc designed for use in low-profile rack-mounted chassis.
[
63
]
It has the connector bracket reversed so it cannot fit in a normal PCI Express socket, but it is pin-compatible and may be inserted if the bracket is removed.
M.2
(formerly known as NGFF)
M-PCIe
brings PCIe 3.0 to mobile devices (such as tablets and smartphones), over the
M-PHY
physical layer.
[
64
]
[
65
]
Serial Attached SCSI
-related ports:
SATA Express
,
U.2
(formerly known as SFF-8639), U.3 use the same port
SlimSAS
(SFF-8654)
SFF-TA-1016 (M-XIO connector)
SFF-TA-1026, SFF-TA-1033
The PCIe slot connector can also carry protocols other than PCIe. Some
9xx series Intel chipsets
support
Serial Digital Video Out
, a proprietary technology that uses a slot to transmit video signals from the host CPU's
integrated graphics
instead of PCIe, using a supported add-in.
The PCIe transaction-layer protocol can also be used over some other interconnects, which are not electrically PCIe:
Thunderbolt
: A royalty-free (as of Thunderbolt 3) interconnect standard by Intel that combines
DisplayPort
and PCIe protocols in a form factor compatible with
Mini DisplayPort
. Thunderbolt 3.0 also combines USB 3.1 and uses the
USB-C
form factor as opposed to Mini DisplayPort.
USB4
is an extension of Thunderbolt 3.0. Thunderbolt 4 and Thunderbolt 5 are
profiles
of USB4 specifying higher levels of mandatory features.
History and revisions
[
edit
]
While in early development, PCIe was initially referred to as
HSI
(for
High Speed Interconnect
), and underwent a name change to
3GIO
(for
3rd Generation I/O
) before finally settling on its
PCI-SIG
name
PCI Express
. A technical working group named the
Arapaho Work Group
(AWG) drew up the standard. For initial drafts, the AWG consisted only of Intel engineers; subsequently, the AWG expanded to include industry partners.
Since, PCIe has undergone several large and smaller revisions, improving on performance and other features.
Comparison table
[
edit
]
PCI Express link performance
[
66
]
[
67
]
[
68
]
Version
Year
Line code
Transfer rate
(per lane)
[
i
]
[
ii
]
Throughput (
GB/s
)
[
i
]
[
iii
]
x1
x2
x4
x8
x16
1.0
2003
NRZ
8b/10b
2.5
GT/s
0.25
0.5
1
2
4
2.0
2007
5.0
GT/s
0.5
1
2
4
8
3.0
2010
128b/130b
8.0
GT/s
0.985
1.969
3.938
7.877
15.754
4.0
2017
16.0
GT/s
1.969
3.938
7.877
15.754
31.508
5.0
2019
32.0
GT/s
3.938
7.877
15.754
31.508
63.015
6.0
2022 (first implemented 2025)
PAM-4
FEC
1b/1b
242B/256B
FLIT
64.0
GT/s
7.563
15.125
30.25
60.5
121
7.0
2025 (specification released)
128.0
GT/s
15.125
30.25
60.5
121
242
8.0
2028
(planned)
256.0
GT/s
30.25
60.5
121
242
484
Notes
1
2
In each direction (each lane is a dual simplex channel).
↑
Transfer rate refers to the encoded serial bit rate; 2.5 GT/s means 2.5
Gbit/s serial data rate.
↑
Throughput indicates the usable bandwidth (i.e. only including the payload, not the 8b/10b, 128b/130b, or 242B/256B encoding overhead). The PCIe 1.0 transfer rate of 2.5
GT/s per lane means a 2.5
Gbit/s serial bit rate; after applying a 8b/10b encoding, this corresponds to a useful throughput of 2.0
Gbit/s = 250
MB/s.
PCI Express 1.0a
[
edit
]
In 2003, PCI-SIG introduced PCIe 1.0a, with a per-lane data rate of 0.25
gigabytes per second
(GB/s) and a transfer rate of 2.5
gigatransfers per second
(GT/s).
Transfer rate is expressed in transfers per second instead of bits per second because the number of transfers includes the overhead bits, which do not provide additional throughput;
[
69
]
PCIe 1.x uses an
8b/10b encoding
scheme, resulting in a 20% (=
2/10) overhead on the raw channel bandwidth.
[
70
]
So in the PCIe terminology, transfer rate refers to the encoded bit rate: 2.5
GT/s is 2.5
Gbit/s on the encoded serial link. This corresponds to 2.0
Gbit/s of pre-coded data or 0.25
GB/s, which is referred to as throughput in PCIe.
PCI Express 1.1
[
edit
]
In 2005, PCI-SIG
[
71
]
introduced PCIe 1.1. This updated specification includes clarifications and several improvements, but is fully compatible with PCI Express 1.0a. No changes were made to the data rate.
PCI Express 2.0
[
edit
]
A PCI Express
2.0 x1 expansion card that provides USB
3.0 connectivity
[
b
]
PCI-SIG
announced the availability of the PCI Express Base 2.0 specification on 15 January 2007.
[
72
]
The PCIe 2.0 standard doubles the transfer rate compared with PCIe 1.0 to 5
GT/s and the per-lane throughput rises from 250
MB/s to 500
MB/s. Consequently, a 16-lane PCIe connector (x16) can support an aggregate throughput of up to 8
GB/s.
PCIe 2.0 motherboard slots are fully
backward compatible
with PCIe v1.x cards. PCIe 2.0 cards are also generally backward compatible with PCIe 1.x motherboards, using the available bandwidth of PCI Express 1.1. Overall, graphic cards or motherboards designed for v2.0 work, with the other being v1.1 or v1.0a.
The PCI-SIG also said that PCIe 2.0 features improvements to the point-to-point data transfer protocol and its software architecture.
[
73
]
Intel
's first PCIe 2.0 capable chipset was the
X38
and boards began to ship from various vendors (
Abit
,
Asus
,
Gigabyte
) as of 21 October 2007.
[
74
]
AMD started supporting PCIe 2.0 with its
AMD 700 chipset series
and nVidia started with the
MCP72
.
[
75
]
All of Intel's prior chipsets, including the
Intel P35
chipset, supported PCIe 1.1 or 1.0a.
[
76
]
Like 1.x, PCIe 2.0 uses an
8b/10b encoding
scheme, therefore delivering, per-lane, an effective 4
Gbit/s max. transfer rate from its 5
GT/s raw data rate.
PCI Express 2.1
[
edit
]
PCI Express 2.1 (with its specification dated 4 March 2009) supports a large proportion of the management, support, and troubleshooting systems planned for full implementation in PCI Express 3.0. However, the speed is the same as PCI Express 2.0. The increase in power from the slot breaks backward compatibility between PCI Express 2.1 cards and some older motherboards with 1.0/1.0a, but most motherboards with PCI Express 1.1 connectors are provided with a BIOS update by their manufacturers through utilities to support backward compatibility of cards with PCIe 2.1.
PCI Express 3.0
[
edit
]
PCI Express 3.0 Base specification revision 3.0 was made available in November 2010, after multiple delays. In August 2007, PCI-SIG announced that PCI Express 3.0 would carry a bit rate of 8
gigatransfers
per second (GT/s), and that it would be backward compatible with existing PCI Express implementations. At that time, it was also announced that the final specification for PCI Express 3.0 would be delayed until Q2 2010.
[
77
]
New features for the PCI Express 3.0 specification included a number of optimizations for enhanced signaling and data integrity, including transmitter and receiver equalization,
PLL
improvements, clock data recovery, and channel enhancements of currently supported topologies.
[
78
]
Following a six-month technical analysis of the feasibility of scaling the PCI Express interconnect bandwidth, PCI-SIG's analysis found that 8 gigatransfers per second could be manufactured in mainstream silicon process technology, and deployed with existing low-cost materials and infrastructure, while maintaining full compatibility (with negligible impact) with the PCI Express protocol stack.
PCI Express
3.0 upgraded the line encoding scheme to 128b/130b from the previous
8b/10b encoding
, reducing the bandwidth overhead from 20% of  PCI Express
2.0 to approximately 1.54% (=
2/130). PCI Express 3.0's 8
GT/s bit rate effectively delivers 985
MB/s per lane, nearly doubling the lane bandwidth relative to PCI Express 2.0.
[
67
]
On 18 November 2010, the PCI-SIG officially published the finalized PCI Express 3.0 specification to its members to build devices based on this new version of PCI Express.
[
79
]
PCI Express 3.1
[
edit
]
In September 2013, PCI Express
3.1 specification was announced for release in late 2013 or early 2014, consolidating various improvements to the published PCI Express
3.0 specification in three areas: power management, performance and functionality.
[
65
]
[
80
]
It was released in November 2014.
[
81
]
PCI Express 4.0
[
edit
]
On 29 November 2011, PCI-SIG preliminarily announced PCI Express 4.0,
[
82
]
providing a 16
GT/s bit rate that doubles the bandwidth provided by PCI Express 3.0 to 31.5
GB/s in each direction for a 16-lane configuration, while maintaining backward and
forward compatibility
in both software support and used mechanical interface.
[
83
]
PCI Express 4.0 specs also bring
OCuLink-2
, an alternative to
Thunderbolt
. OCuLink version 2 has up to 16
GT/s (16
GB/s total for x8 lanes),
[
58
]
while the maximum bandwidth of a Thunderbolt 3 link is 5
GB/s.
At the
2016 PCI-SIG Developers Conference
,
Cadence
,
PLDA
, and
Synopsys
demonstrated their development of PCIe 4.0 physical layer, controller, switch, and other IP blocks.
[
84
]
Mellanox Technologies
announced the first 100
Gbit/s network adapter with PCIe 4.0 on 15 June 2016,
[
85
]
and the first 200
Gbit/s network adapter with PCIe 4.0 on 10 November 2016.
[
86
]
In August 2016, Synopsys presented a test setup with FPGA clocking a lane to PCIe 4.0 speeds at the
Intel Developer Forum
. Their IP has been licensed to several firms planning to present their chips and products at the end of 2016.
[
87
]
On the IEEE Hot Chips Symposium in August 2016
IBM
announced the first CPU with PCIe 4.0 support,
POWER9
.
[
88
]
[
89
]
PCI-SIG officially announced the release of the final PCI Express 4.0 specification on 8 June 2017.
[
90
]
The spec includes improvements in flexibility, scalability, and lower-power.
On 5 December 2017 IBM announced the first system with PCIe 4.0 slots, Power AC922.
[
91
]
[
92
]
NETINT Technologies introduced the first
NVMe
SSD based on PCIe 4.0 on 17 July 2018, ahead of Flash Memory Summit 2018
[
93
]
AMD
announced on 9 January 2019 its upcoming
Zen 2
-based processors and X570 chipset would support PCIe 4.0.
[
94
]
AMD had hoped to enable partial support for older chipsets, but instability caused by motherboard traces not conforming to PCIe 4.0 specifications made that impossible.
[
95
]
[
96
]
Intel released their first mobile CPUs with PCI Express 4.0 support in mid-2020, as a part of the
Tiger Lake
microarchitecture.
[
97
]
PCI Express 5.0
[
edit
]
Three PCIe 5.0 x16 (first and third slots at x16, fourth slot at x8 throughput) and two PCIe 4.0 x16 slots (second slot at x4, fifth slot at x8 throughput) on a 2023 workstation mainboard.
In June 2017, PCI-SIG announced the PCI Express 5.0 preliminary specification.
[
90
]
Bandwidth was expected to increase to 32
GT/s, yielding 63
GB/s
in each direction in a 16-lane configuration. The draft spec was expected to be standardized in 2019.
[
citation needed
]
Initially,
25.0 GT/s
was also considered for technical feasibility.
On 7 June 2017 at PCI-SIG DevCon, Synopsys recorded the first demonstration of PCI Express 5.0 at 32
GT/s.
[
98
]
On 31 May 2018, PLDA announced the availability of their XpressRICH5 PCIe 5.0 Controller IP based on draft 0.7 of the PCIe 5.0 specification on the same day.
[
99
]
[
100
]
On 10 December 2018, the PCI SIG released version 0.9 of the PCIe 5.0 specification to its members,
[
101
]
and on 17 January 2019, PCI SIG announced the version 0.9 had been ratified, with version 1.0 targeted for release in the first quarter of 2019.
[
102
]
On 29 May 2019, PCI-SIG officially announced the release of the final PCI Express 5.0 specification.
[
103
]
The PCI Express 5.0 retained
backward compatibility
with previous versions of PCI Express specifications.
On 20 November 2019,
Jiangsu Huacun
presented the first PCIe 5.0 Controller HC9001 in a 12
nm manufacturing process
[
104
]
and production started in 2020.
On 17 August 2020, IBM announced the
Power10
processor with PCIe 5.0 and up to 32 lanes per single-chip module (SCM) and up to 64 lanes per double-chip module (DCM).
[
105
]
On 9 September 2021, IBM announced the Power E1080 Enterprise server with planned availability date 17 September.
[
106
]
It can have up to 16 Power10 SCMs with maximum of 32 slots per system which can act as PCIe 5.0 x8 or PCIe 4.0 x16.
[
107
]
Alternatively they can be used as PCIe 5.0 x16 slots for optional optical CXP converter adapters connecting to external PCIe expansion drawers.
On 27 October 2021, Intel announced the 12th Gen Intel Core CPU family, the world's first consumer x86-64 processors with PCIe 5.0 (up to 16 lanes) connectivity.
[
108
]
On 22 March 2022, Nvidia announced Nvidia Hopper GH100 GPU, the world's first PCIe 5.0 GPU.
[
109
]
On 23 May 2022, AMD announced its Zen 4 architecture with support for up to 24 lanes of PCIe 5.0 connectivity on consumer platforms and 128 lanes on server platforms.
[
110
]
[
111
]
PCI Express 6.0
[
edit
]
On 18 June 2019, PCI-SIG announced the development of PCI Express 6.0 specification. Bandwidth is expected to increase to 64
GT/s, yielding 128
GB/s in each direction in a 16-lane configuration, with a target release date of 2021.
[
112
]
The new standard uses 4-level
pulse-amplitude modulation
(PAM-4) with a low-latency
forward error correction
(FEC) in place of
non-return-to-zero
(NRZ) modulation.
[
113
]
Unlike previous PCI Express versions, forward error correction is used to increase data integrity and PAM-4 is used as line code so that two bits are transferred per transfer. With 64
GT/s data transfer rate (raw bit rate), up to 121
GB/s in each direction is possible in x16 configuration.
[
112
]
On 24 February 2020, the PCI Express 6.0 revision 0.5 specification (a "first draft" with all architectural aspects and requirements defined) was released.
[
114
]
On 5 November 2020, the PCI Express 6.0 revision 0.7 specification (a "complete draft" with electrical specifications validated via test chips) was released.
[
115
]
On 6 October 2021, the PCI Express 6.0 revision 0.9 specification (a "final draft") was released.
[
116
]
On 11 January 2022, PCI-SIG officially announced the release of the final PCI Express 6.0 specification.
[
117
]
The PCI Express 6.0 retained
backward compatibility
with previous versions of PCI Express specifications.
PAM-4
coding results in a vastly higher
bit error rate
(BER) of 10
−6
(vs. 10
−12
previously), so in place of 128b/130b encoding, a 3-way interlaced
forward error correction
(FEC) is used in addition to
cyclic redundancy check
(CRC). A fixed 256 byte
Flow Control Unit
(FLIT) block carries 242 bytes of data, which includes variable-sized transaction level packets (TLP) and data link layer payload (DLLP); remaining 14 bytes are reserved for 8-byte CRC and 6-byte FEC.
[
118
]
[
119
]
3-way
Gray code
is used in PAM-4/FLIT mode to reduce error rate; the interface does not switch to NRZ and 128/130b encoding even when retraining to lower data rates.
[
120
]
[
121
]
PCIe 6.0 hardware was not launched until August 2025,
[
122
]
roughly three years after the release of the final specifications and shortly after the publication of the PCIe 7.0 specifications.
[
123
]
The delay was described as unprecedented, with
PCWorld
noting that for many years PCIe 6.0 existed "solely on paper".
[
124
]
PCI Express 7.0
[
edit
]
On 21 June 2022, PCI-SIG announced the development of PCI Express 7.0 specification.
[
125
]
It will deliver 128
GT/s raw bit rate and up to 242
GB/s per direction in x16 configuration, using the same
PAM4
signaling as version 6.0. Doubling of the data rate will be achieved by fine-tuning channel parameters to decrease signal losses and improve power efficiency, but signal integrity is expected to be a challenge. The specification is expected to be finalized in 2025.
On 3 April 2024, the PCI Express 7.0 revision 0.5 specification (a "first draft") was released.
[
126
]
On 17 January 2025, PCI-SIG announced the release of PCIe 7.0 specification version 0.7 (a "complete draft").
[
127
]
On 19 March 2025, PCI-SIG announced the release of PCIe 7.0 specification version 0.9 (a "final draft"); planned final release is still in 2025.
[
128
]
The following main points were formulated as objectives of the new standard:
Delivering 128 GT/s raw bit rate and up to 512 GB/s bi-directionally via x16 configuration
Utilizing PAM4 (Pulse Amplitude Modulation with 4 levels) signaling
Focusing on the channel parameters and reach
Improving power efficiency
Continuing to deliver the low-latency and high-reliability targets
Maintaining backwards compatibility with all previous generations of PCIe technology
On 11 June 2025, PCI-SIG officially announced the release of the final PCI Express 7.0 specification.
[
129
]
At its release, PCI-SIG commented that it did not see the PCIe 7.0 coming to the PC market for some time. Instead the interface is initially targeted at cloud computing, 800-gigabit Ethernet, and artificial intelligence applications.
[
124
]
PCI Express 8.0
[
edit
]
On 5 August 2025, PCI-SIG announced the development of PCI Express 8.0. The specification is planned for release by year 2028. It will deliver double the speed of the previous version, 256.0 GT/s raw bit rate and up to 1 TB/s bi-directionally via x16 configuration.
[
130
]
Extensions and future directions
[
edit
]
Some vendors offer PCIe over fiber products,
[
131
]
[
132
]
[
133
]
with active optical cables (AOC) for PCIe switching at increased distance in PCIe expansion drawers,
[
134
]
[
107
]
or in specific cases where transparent PCIe bridging is preferable to using a more mainstream standard (such as
InfiniBand
or
Ethernet
) that may require additional software to support it.
Thunderbolt
was co-developed by
Intel
and
Apple
as a general-purpose high speed interface combining a logical PCIe link with
DisplayPort
and was originally intended as an all-fiber interface, but due to early difficulties in creating a consumer-friendly fiber interconnect, nearly all implementations are copper systems. A notable exception, the
Sony VAIO Z
VPC-Z2, uses a nonstandard USB port with an optical component to connect to an outboard PCIe display adapter. Apple has been the primary driver of Thunderbolt adoption through 2011, though several other vendors
[
135
]
have announced new products and systems featuring Thunderbolt. Thunderbolt 3 forms the basis of the
USB4
standard.
Mobile PCIe
specification (abbreviated to
M-PCIe
) allows PCI Express architecture to operate over the
MIPI Alliance
's
M-PHY
physical layer technology. Building on top of already existing widespread adoption of M-PHY and its low-power design, Mobile PCIe lets mobile devices use PCI Express.
[
136
]
iPhone
is one example that utilizing integrated
NVMe
storage with M-PCIe.
Draft process
[
edit
]
There are 5 primary releases/checkpoints in a PCI-SIG specification:
[
137
]
Draft 0.3 (Concept): this release may have few details, but outlines the general approach and goals.
Draft 0.5 (First draft): this release has a complete set of architectural requirements and must fully address the goals set out in the 0.3 draft.
Draft 0.7 (Complete draft): this release must have a complete set of functional requirements and methods defined, and no new functionality may be added to the specification after this release. Before the release of this draft, electrical specifications must have been validated via test silicon.
Draft 0.9 (Final draft): this release allows PCI-SIG member companies to perform an internal review for intellectual property, and no functional changes are permitted after this draft.
1.0 (Final release): this is the final and definitive specification, and any changes or enhancements are through Errata documentation and Engineering Change Notices (ECNs) respectively.
Historically, the earliest adopters of a new PCIe specification generally begin designing with the Draft 0.5 as they can confidently build up their application logic around the new bandwidth definition and often even start developing for any new protocol features. At the Draft 0.5 stage, however, there is still a strong likelihood of changes in the actual PCIe protocol layer implementation, so designers responsible for developing these blocks internally may be more hesitant to begin work than those using interface IP from external sources.
Hardware protocol summary
[
edit
]
The PCIe link is built around dedicated unidirectional couples of serial (1-bit), point-to-point connections known as
lanes
. This is in sharp contrast to the earlier PCI connection, which is a bus-based system where all the devices share the same bidirectional, 32-bit or 64-bit parallel bus.
PCI Express is a
layered protocol
, consisting of a
transaction layer
, a
data link layer
, and a
physical layer
. The Data Link Layer is subdivided to include a
media access control
(MAC) sublayer. The Physical Layer is subdivided into logical and electrical sublayers. The Physical logical-sublayer contains a physical coding sublayer (PCS). The terms are borrowed from the
IEEE 802
networking protocol model.
Physical layer
[
edit
]
Connector pins and lengths
Lanes
Pins
[
138
]
Length in mm (in)
Board connector
Connector slot
Total
Variable
Total
Variable
Total
Variable
x1
2×18=36
2×7=14
20.3 (0.8)
7.2 (0.28)
25 (1.0)
7.65 (0.30)
x4
2×32=64
2×21=42
34.3 (1.4)
21.2 (0.8)
39 (1.5)
21.65 (0.85)
x8
2×49=98
2×38=76
51.3 (2.0)
38.2 (1.5)
56 (2.2)
38.65 (1.52)
x16
2×82=164
2×71=142
84.3 (3.3)
71.2 (2.8)
89 (3.5)
71.65 (2.82)
An open-end PCI Express x1 connector lets longer cards that use more lanes be plugged while operating at x1 speeds.
The PCIe Physical Layer (
PHY
,
PCIEPHY
,
PCI Express PHY
, or
PCIe PHY
) specification is divided into two sub-layers, corresponding to electrical and logical specifications. The logical sublayer is sometimes further divided into a MAC sublayer and a PCS, although this division is not formally part of the PCIe specification. A specification published by Intel, the PHY Interface for PCI Express (PIPE),
[
139
]
defines the MAC/PCS functional partitioning and the interface between these two sub-layers. The PIPE specification also identifies the
physical media attachment
(PMA) layer, which includes the
SerDes
(serializer/deserializer) and other analog circuitry; however, since SerDes implementations vary greatly among
ASIC
vendors, PIPE does not specify an interface between the PCS and PMA.
At the electrical level, each lane consists of two unidirectional
differential pairs
operating at 2.5, 5, 8, 16 or 32
Gbit
/s, depending on the negotiated capabilities. Transmit and receive are separate differential pairs, for a total of four data wires per lane.
A connection between any two PCIe devices is known as a
link
, and is built up from a collection of one or more
lanes
. All devices must minimally support single-lane (x1) link. Devices may optionally support wider links composed of up to 32 lanes.
[
140
]
[
141
]
This allows for very good compatibility in two ways:
A PCIe card physically fits (and works correctly) in any slot that is at least as large as it is (e.g., a x1 sized card works in any sized slot);
A slot of a large physical size (e.g., x16) can be wired electrically with fewer lanes (e.g., x1, x4, x8, or x12) as long as it provides the ground connections required by the larger physical slot size.
In both cases, PCIe negotiates the highest mutually supported number of lanes. Many graphics cards, motherboards and
BIOS
versions are verified to support x1, x4, x8 and x16 connectivity on the same connection.
The width of a PCIe connector is 8.8
mm, while the height is 11.25
mm, and the length is variable. The fixed section of the connector is 11.65
mm in length and contains two rows of 11 pins each (22 pins total), while the length of the other section is variable depending on the number of lanes. The pins are spaced at 1
mm intervals, and the thickness of the card going into the connector is 1.6
mm.
[
142
]
[
143
]
Data transmission
[
edit
]
PCIe sends all control messages, including interrupts, over the same links used for data. The serial protocol can never be blocked, so latency is still comparable to conventional PCI, which has dedicated interrupt lines. When the problem of IRQ sharing of pin based interrupts is taken into account and the fact that message signaled interrupts (MSI) can bypass an I/O APIC and be delivered to the CPU directly, MSI performance ends up being substantially better.
[
144
]
Data transmitted on multiple-lane links is interleaved, meaning that each successive byte is sent down successive lanes. The PCIe specification refers to this interleaving as
data striping
. While requiring significant hardware complexity to synchronize (or
deskew
) the incoming striped data, striping can significantly reduce the latency of the
n
th byte on a link. While the lanes are not tightly synchronized, there is a limit to the
lane to lane skew
of 20/8/6
ns for 2.5/5/8
GT/s so the hardware buffers can re-align the striped data.
[
145
]
Due to padding requirements, striping may not necessarily reduce the latency of small data packets on a link.
As with other high data rate serial transmission protocols, the clock is
embedded
in the signal. At the physical level, PCI Express 2.0 utilizes the
8b/10b encoding
scheme
[
67
]
(line code) to ensure that strings of consecutive identical digits (zeros or ones) are limited in length. This coding was used to prevent the receiver from losing track of where the bit edges are. In this coding scheme every eight (uncoded) payload bits of data are replaced with 10 (encoded) bits of transmit data, causing a 20% overhead in the electrical bandwidth. To improve the available bandwidth, PCI Express version 3.0 instead uses
128b/130b
encoding (1.54% overhead).
Line encoding
limits the run length of identical-digit strings in data streams and ensures the receiver stays synchronised to the transmitter via
clock recovery
.
A desirable balance (and therefore
spectral density
) of 0 and 1 bits in the data stream is achieved by
XORing
a known
binary polynomial
as a "
scrambler
" to the data stream in a feedback topology. Because the scrambling polynomial is known, the data can be recovered by applying the XOR a second time. Both the scrambling and descrambling steps are carried out in hardware.
Dual simplex in PCIe means there are two simplex channels on every PCIe lane. Simplex means communication is only possible in one direction. By having two simplex channels, two-way communication is made possible. One differential pair is used for each channel.
[
146
]
[
1
]
[
147
]
Data link layer
[
edit
]
The data link layer performs three vital services for the PCIe link:
sequence the transaction layer packets (TLPs) that are generated by the transaction layer,
ensure reliable delivery of TLPs between two endpoints via an acknowledgement protocol (
ACK
and
NAK
signaling) that explicitly requires replay of unacknowledged/bad TLPs,
initialize and manage flow control credits
On the transmit side, the data link layer generates an incrementing sequence number for each outgoing TLP. It serves as a unique identification tag for each transmitted TLP, and is inserted into the header of the outgoing TLP. A 32-bit
cyclic redundancy check
code (known in this context as Link CRC or LCRC) is also appended to the end of each outgoing TLP.
On the receive side, the received TLP's LCRC and sequence number are both validated in the link layer. If either the LCRC check fails (indicating a data error), or the sequence-number is out of range (non-consecutive from the last valid received TLP), then the bad TLP, as well as any TLPs received after the bad TLP, are considered invalid and discarded. The receiver sends a negative acknowledgement message (NAK) with the sequence-number of the invalid TLP, requesting re-transmission of all TLPs forward of that sequence-number. If the received TLP passes the LCRC check and has the correct sequence number, it is treated as valid. The link receiver increments the sequence-number (which tracks the last received good TLP), and forwards the valid TLP to the receiver's transaction layer. An ACK message is sent to remote transmitter, indicating the TLP was successfully received (and by extension, all TLPs with past sequence-numbers.)
If the transmitter receives a NAK message, or no acknowledgement (NAK or ACK) is received until a timeout period expires, the transmitter must retransmit all TLPs that lack a positive acknowledgement (ACK). Barring a persistent malfunction of the device or transmission medium, the link-layer presents a reliable connection to the transaction layer, since the transmission protocol ensures delivery of TLPs over an unreliable medium.
In addition to sending and receiving TLPs generated by the transaction layer, the data-link layer also generates and consumes data link layer packets (DLLPs). ACK and NAK signals are communicated via DLLPs, as are some power management messages and flow control credit information (on behalf of the transaction layer).
In practice, the number of in-flight, unacknowledged TLPs on the link is limited by two factors: the size of the transmitter's replay buffer (which must store a copy of all transmitted TLPs until the remote receiver ACKs them), and the flow control credits issued by the receiver to a transmitter. PCI Express requires all receivers to issue a minimum number of credits, to guarantee a link allows sending PCIConfig TLPs and message TLPs.
Transaction layer
[
edit
]
PCI Express implements split transactions (transactions with request and response separated by time), allowing the link to carry other traffic while the target device gathers data for the response.
PCI Express uses credit-based flow control. In this scheme, a device advertises an initial amount of credit for each received buffer in its transaction layer. The device at the
opposite end of the link, when sending transactions to this device, counts the number of credits each TLP consumes from its account. The sending device may only transmit a TLP when doing so does not make its consumed credit count exceed its credit limit. When the receiving device finishes processing the TLP from its buffer, it signals a return of credits to the sending device, which increases the credit limit by the restored amount. The credit counters are modular counters, and the comparison of consumed credits to credit limit requires
modular arithmetic
. The advantage of this scheme (compared to other methods such as wait states or handshake-based transfer protocols) is that the latency of credit return does not affect performance, provided that the credit limit is not encountered. This assumption is generally met if each device is designed with adequate buffer sizes.
PCIe 1.x is often quoted to support a data rate of 250
MB/s in each direction, per lane. This figure is a calculation from the physical signaling rate (2.5
gigabaud
) divided by the encoding overhead (10 bits per byte). This means a sixteen lane (x16) PCIe card would then be theoretically capable of 16×250
MB/s = 4
GB/s in each direction. While this is correct in terms of data bytes, more meaningful calculations are based on the usable data payload rate, which depends on the profile of the traffic, which is a function of the high-level (software) application and intermediate protocol levels.
Like other high data rate serial interconnect systems, PCIe has a protocol and processing overhead due to the additional transfer robustness (CRC and acknowledgements). Long continuous unidirectional transfers (such as those typical in high-performance storage controllers) can approach >95% of PCIe's raw (lane) data rate. These transfers also benefit the most from increased number of lanes (x2, x4, etc.) But in more typical applications (such as a
USB
or
Ethernet
controller), the traffic profile is characterized as short data packets with frequent enforced acknowledgements.
[
148
]
This type of traffic reduces the efficiency of the link, due to overhead from packet parsing and forced interrupts (either in the device's host interface or the PC's CPU). Being a protocol for devices connected to the same
printed circuit board
, it does not require the same tolerance for transmission errors as a protocol for communication over longer distances, and thus, this loss of efficiency is not particular to PCIe.
Efficiency of the link
[
edit
]
As for any network-like communication links, some of the raw bandwidth is consumed by protocol overhead:
[
149
]
A PCIe 1.x lane for example offers a data rate on top of the physical layer of 250
MB/s (simplex). This is due to a 2.5
GT/s bit rate multiplied by the efficiency of the 8b/10b line code (see
#Comparison table
). This is not the payload bandwidth but the physical layer bandwidth – a PCIe lane has to carry additional information for full functionality.
[
149
]
Gen 2 Transaction Layer Packet
[
149
]
:
3
Layer
PHY
Data Link Layer
Transaction
Data Link Layer
PHY
Data
Start
Sequence
Header
Payload
ECRC
LCRC
End
Size (Bytes)
1
2
12 or 16
0 to 4096
4 (optional)
4
1
The Gen2 overhead is then 20, 24, or 28 bytes per transaction.
Gen 3 Transaction Layer Packet
[
149
]
:
3
Layer
PHY
Data Link Layer
Transaction Layer
Data Link Layer
Data
Start
Sequence
Header
Payload
ECRC
LCRC
Size (Bytes)
4
2
12 or 16
0 to 4096
4 (optional)
4
The Gen3 overhead is then 22, 26 or 30 bytes per transaction.
The
Packet Efficiency =
⁠
Payload
/
Payload + Overhead
⁠
for a 128 byte payload is 86%, and 98% for a 1024 byte payload. For small accesses like register settings (4 bytes), the efficiency drops as low as 16%. That said, most PCIe config registers reside in a DMA region mapped to the CPU's control registers and require no bus access.
[
citation needed
]
The maximum payload size (MPS) is set on all devices based on smallest maximum on any device in the chain. If one device has an MPS of 128 bytes,
all
devices of the tree must set their MPS to 128 bytes. In this case the bus will have a maximum efficiency of 86% for writes.
[
149
]
:
3
Applications
[
edit
]
Asus
Nvidia GeForce GTX 650 Ti, a PCI Express 3.0 x16 graphics card
The
Nvidia
GeForce GTX 1070, a PCI Express 3.0 x16 Graphics card
Intel
82574L Gigabit Ethernet
NIC
, a PCI Express x1 card
A
Marvell
-based
SATA 3.0
controller, as a PCI Express x1 card
PCI Express operates in consumer, server, and industrial applications, as a motherboard-level interconnect (to link motherboard-mounted peripherals), a passive backplane interconnect and as an
expansion card
interface for add-in boards.
In virtually all modern (
as of 2012
[
update
]
) PCs, from consumer laptops and desktops to enterprise servers, the PCIe bus serves as the primary motherboard-level interconnect, connecting the host system-processor with both integrated peripherals (surface-mounted ICs) and add-on peripherals (expansion cards). In some of these systems, the PCIe bus co-exists with one or more legacy PCI buses, for backward compatibility with the large body of legacy PCI peripherals.
As of 2013
[
update
]
, PCI Express has replaced
AGP
as the default interface for graphics cards on new systems. Almost all models of
graphics cards
released since 2010 by
AMD
(ATI) and
Nvidia
use PCI Express. AMD, Nvidia, and Intel have released motherboard chipsets that support as many as four PCIe x16 slots, allowing tri-GPU and quad-GPU card configurations.
External GPUs
[
edit
]
Theoretically, external PCIe could give a notebook the graphics power of a desktop, by connecting a notebook with any PCIe desktop video card (enclosed in its own external housing, with a power supply and cooling); this is possible with an ExpressCard or
Thunderbolt
interface. An ExpressCard interface provides
bit rates
of 5
Gbit/s (0.5
GB/s throughput), whereas a Thunderbolt interface provides bit rates of up to 40
Gbit/s (5
GB/s throughput).
In 2006,
Nvidia
developed the
Quadro Plex
external PCIe family of
GPUs
that can be used for advanced graphic applications for the professional market.
[
150
]
These video cards require a PCI Express x8 or x16 slot for the host-side card, which connects to the Plex via a
VHDCI
carrying eight PCIe lanes.
[
151
]
In 2008, AMD announced the
ATI XGP
technology, based on a proprietary cabling system that is compatible with PCIe x8 signal transmissions.
[
152
]
This connector is available on the Fujitsu Amilo and the Acer Ferrari One notebooks. Fujitsu launched their AMILO GraphicBooster enclosure for XGP soon thereafter.
[
153
]
Around 2010 Acer launched the Dynavivid graphics dock for XGP.
[
154
]
In 2010, external card hubs were introduced that can connect to a laptop or desktop through a PCI ExpressCard slot. These hubs can accept full-sized graphics cards. Examples include MSI GUS,
[
155
]
Village Instrument's ViDock,
[
156
]
the Asus
XG Station
, Bplus PE4H V3.2 adapter,
[
157
]
as well as more improvised DIY devices.
[
158
]
However such solutions are limited by the size (often only x1) and version of the available PCIe slot on a laptop.
The Intel Thunderbolt interface has provided a new option to connect with a PCIe card externally. Magma has released the ExpressBox 3T, which can hold up to three PCIe cards (two at x8 and one at x4).
[
159
]
MSI also released the Thunderbolt GUS II, a PCIe chassis dedicated for video cards.
[
160
]
Other products such as the Sonnet's Echo Express
[
161
]
and mLogic's mLink are Thunderbolt PCIe chassis in a smaller form factor.
[
162
]
In 2017, more fully featured external card hubs were introduced, such as the Razer Core, which has a full-length PCIe x16 interface.
[
163
]
Storage devices
[
edit
]
An
OCZ
RevoDrive
SSD
, a full-height x4 PCI Express card
See also:
SATA Express
and
NVMe
The PCI Express protocol can be used as data interface to
flash memory
devices, such as
memory cards
and
solid-state drives
(SSDs).
The
XQD card
is a memory card format utilizing PCI Express, developed by the CompactFlash Association, with transfer rates of up to 1
GB/s.
[
164
]
Many high-performance, enterprise-class SSDs are designed as PCI Express
RAID controller
cards.
[
citation needed
]
Before NVMe was standardized, many of these cards utilized proprietary interfaces and custom drivers to communicate with the operating system; they had much higher transfer rates (over 1
GB/s) and IOPS (over one million I/O operations per second) when compared to Serial ATA or
SAS
drives.
[
quantify
]
[
165
]
[
166
]
For example, in 2011 OCZ and Marvell co-developed a native PCI Express solid-state drive controller for a PCI Express 3.0 x16 slot with maximum capacity of 12
TB and a performance of to 7.2
GB/s sequential transfers and up to 2.52
million IOPS in random transfers.
[
167
]
[
relevant?
]
SATA Express
was an interface for connecting SSDs through SATA-compatible ports, optionally providing multiple PCI Express lanes as a pure PCI Express connection to the attached storage device.
[
168
]
M.2
is a specification for internally mounted computer
expansion cards
and associated connectors, which can use up to four PCI Express lanes.
[
169
]
PCI Express storage devices can implement both
AHCI
logical interface for backward compatibility, and
NVM Express
logical interface for much faster I/O operations provided by utilizing internal parallelism offered by such devices. Enterprise-class SSDs can also implement
SCSI over PCI Express
.
[
170
]
Cluster interconnect
[
edit
]
Certain
data-center
applications (such as large
computer clusters
) require the use of fiber-optic interconnects due to the distance limitations inherent in copper cabling. Typically, a network-oriented standard such as Ethernet or
Fibre Channel
suffices for these applications, but in some cases the overhead introduced by
routable
protocols is undesirable and a lower-level interconnect, such as
InfiniBand
,
RapidIO
, or
NUMAlink
is needed. Local-bus standards such as PCIe and
HyperTransport
can in principle be used for this purpose,
[
171
]
but
as of 2015
[
update
]
, solutions are only available from niche vendors such as
Dolphin ICS
, and TTTech Auto.
Competing protocols
[
edit
]
PCIe 1.0 initially competed with
PCI-X 2.0
, with both specifications being ratified in 2003 and offering roughly the same maximum bandwidth (~4
GB/s). By 2005, however, PCIe emerged as the dominant technology.
Other communications standards based on high bandwidth serial architectures include
InfiniBand
,
RapidIO
,
HyperTransport
,
Intel QuickPath Interconnect
, the
Mobile Industry Processor Interface
(MIPI), and
NVLink
. Differences are based on the trade-offs between flexibility and extensibility vs latency and overhead. For example, making the system hot-pluggable, as with Infiniband but not PCI Express, requires that software track network topology changes.
[
citation needed
]
Another example is making the packets shorter to decrease latency (as is required if a bus must operate as a memory interface). Smaller packets mean packet headers consume a higher percentage of the packet, thus decreasing the effective bandwidth. Examples of bus protocols designed for this purpose are RapidIO and HyperTransport.
[
citation needed
]
PCI Express falls somewhere in the middle,
[
clarification needed
]
targeted by design as a system interconnect (
local bus
) rather than a device interconnect or routed network protocol. Additionally, its design goal of software transparency constrains the protocol and raises its latency somewhat.
[
citation needed
]
Delays in PCIe 4.0 implementations led to the
Gen-Z
consortium, the CCIX effort and an open
Coherent Accelerator Processor Interface
(CAPI) all being announced by the end of 2016.
[
172
]
On 11 March 2019, Intel presented
Compute Express Link
(CXL), a new interconnect bus, based on the PCI Express 5.0 physical layer infrastructure. The initial promoters of the CXL specification included:
Alibaba
,
Cisco
,
Dell EMC
,
Facebook
,
Google
,
HPE
,
Huawei
,
Intel
and
Microsoft
.
[
173
]
Integrators list
[
edit
]
The PCI-SIG Integrators List lists products made by PCI-SIG member companies that have passed compliance testing. The list include switches, bridges, NICs, SSDs, etc.
[
174
]
See also
[
edit
]
Electronics portal
Active State Power Management
(ASPM)
Peripheral Component Interconnect
(PCI)
PCI configuration space
PCI-X
(PCI Extended)
PCI/104-Express
PCIe/104
Root complex
Serial Digital Video Out
(SDVO)
List of device bit rates §
Main buses
UCIe
Compute Express Link
(CXL)
Notes
[
edit
]
↑
Switches can create multiple endpoints out of one to allow sharing it with multiple devices.
↑
The card's
Serial ATA power connector
is present because the USB
3.0 ports require more power than the PCI Express bus can supply. More often, a
4-pin Molex power connector
is used.
References
[
edit
]
1
2
IBM Power 770 and 780 Technical Overview and Introduction
. IBM Redbooks. 6 June 2013.
ISBN
978-0-7384-5121-3
.
↑
Mayhew, D.; Krishnan, V. (August 2003). "PCI express and advanced switching: Evolutionary path to building next generation interconnects".
11th Symposium on High Performance Interconnects, 2003. Proceedings
. pp.
21–
29.
doi
:
10.1109/CONECT.2003.1231473
.
ISBN
0-7695-2012-X
.
S2CID
7456382
.
↑
"Definition of PCI Express"
.
PCMag
.
↑
Zhang, Yanmin; Nguyen, T Long (June 2007).
"Enable PCI Express Advanced Error Reporting in the Kernel"
(PDF)
.
Proceedings of the Linux Symposium
. Fedora project. Archived from
the original
(PDF)
on 10 March 2016
. Retrieved
8 May
2012
.
↑
https://www.hyperstone.com
Flash Memory Form Factors – The Fundamentals of Reliable Flash Storage, Retrieved 19 April 2018
1
2
3
Ravi Budruk (21 August 2007).
"PCI Express Basics"
.
PCI-SIG
. Archived from
the original
(PDF)
on 15 July 2014
. Retrieved
15 July
2014
.
↑
"What are PCIe Slots and Their Uses"
. PC Guide 101. 18 May 2021
. Retrieved
21 June
2021
.
↑
Wilson, Tracy V. (17 August 2005).
"How PCI Express Works"
.
How Stuff Works
.
Archived
from the original on 3 December 2009
. Retrieved
7 December
2009
.
↑
"4.2.4.9. Link Width and Lane Sequence Negotiation",
PCI Express Base Specification, Revision 2.1.
, 4 March 2009
1
2
3
"PCI Express Architecture Frequently Asked Questions"
. PCI-SIG. Archived from
the original
on 13 November 2008
. Retrieved
23 November
2008
.
↑
"PCI Express Bus"
.
Interface bus
. Archived from
the original
on 8 December 2007
. Retrieved
12 June
2010
.
↑
32 lanes are defined by the
PCIe Base Specification
up to PCIe 5.0 but there's no card standard in the
PCIe Card Electromechanical Specification
and that lane number was never implemented.
↑
"PCI Express – An Overview of the PCI Express Standard"
.
Developer Zone
. National Instruments. 13 August 2009. Archived from
the original
on 5 January 2010
. Retrieved
7 December
2009
.
↑
Qazi, Atif.
"What are PCIe Slots?"
.
PC Gear Lab
. Archived from
the original
on 3 August 2020
. Retrieved
8 April
2020
.
↑
Do, Vanessa (11 September 2025).
"Rethinking AI Infrastructure: The Rise Of PCIe Switches"
.
Semiconductor Engineering
.
↑
"New PCIe Form Factor Enables Greater PCIe SSD Adoption"
.
NVM Express
. 12 June 2012.
Archived
from the original on 6 September 2015.
↑
"Memblaze PBlaze4 AIC NVMe SSD Review"
.
StorageReview
. 21 December 2015.
1
2
PCI Express Card Electromechanical Specification Revision 4.0, Version 0.9=
. November 2018.
(PCIe_CEM_SPEC_R4_V9_12072018_NCB.pdf)
↑
Fulton, Kane (20 July 2015).
"19 graphics cards that shaped the future of gaming"
.
TechRadar
.
↑
Leadbetter, Richard (16 September 2020).
"Nvidia GeForce RTX 3080 review: welcome to the next level"
.
Eurogamer
.
↑
Discuss, btarunr (6 January 2023).
"ASUS × Noctua RTX 4080 Graphics Card is 5 Slots Thick, We Go Hands-on"
.
TechPowerUp
. Retrieved
19 September
2024
.
↑
"Sapphire Radeon RX 5700 XT Pulse Review | bit-tech.net"
.
bit-tech.net
. Retrieved
26 August
2019
.
↑
"AMD Radeon RX 5700 XT 8GB GDDR6 THICC II – RX-57XT8DFD6"
.
xfxforce.com
. Archived from
the original
on 1 September 2019
. Retrieved
25 August
2019
.
↑
"ROG Strix GeForce RTX 3080 OC Edition 10GB GDDR6X
|
Graphics Cards"
.
rog.asus.com
.
↑
"What is the A side, B side configuration of PCI cards"
.
Frequently Asked Questions
. Adex Electronics. 1998. Archived from
the original
on 2 November 2011
. Retrieved
24 October
2011
.
1
2
PCI Express Card Electromechanical Specification Revision 2.0
↑
"PCI Express Card Electromechanical Specification Revision 4.0, Version 1.0 (Clean)"
.
↑
"L1 PM Substates with CLKREQ, Revision 1.0a"
(PDF)
. PCI-SIG. Archived from
the original
(PDF)
on 4 December 2018
. Retrieved
8 November
2018
.
↑
"Emergency Power Reduction Mechanism with PWRBRK Signal ECN"
(PDF)
. PCI-SIG. Archived from
the original
(PDF)
on 9 November 2018
. Retrieved
8 November
2018
.
↑
"Where Does PCIe Cable Go?"
. 16 January 2022
. Retrieved
10 June
2022
.
↑
PCI Express Card Electromechanical Specification Revision 1.1
↑
Schoenborn, Zale (2004),
Board Design Guidelines for PCI Express Architecture
(PDF)
, PCI-SIG, pp.
19–
21,
archived
(PDF)
from the original on 27 March 2016
↑
PCI Express Base Specification, Revision 1.1
Page 332
1
2
"Mini-Fit PCI Express* Wire to Board Connector System"
(PDF)
. Retrieved
4 December
2020
.
↑
PCI Express x16 Graphics 150W-ATX Specification Revision 1.0
↑
PCI Express 225 W/300 W High Power Card Electromechanical Specification Revision 1.0
↑
PCI Express Card Electromechanical Specification Revision 3.0
↑
Yun Ling (16 May 2008).
"PCIe Electromechanical Updates"
. Archived from
the original
on 5 November 2015
. Retrieved
7 November
2015
.
↑
"12VHPWR Sideband Allocation and Requirements - PCIe 5.x ECN"
. PCI SIG. 12 May 2022.
↑
"12V-2x6 Connector Updates to PCIe Base 6.0 - PCIe 6.x ECN"
. PCI SIG. 31 August 2023.
This ECN defines Connector Type encodings for the new 12V-2x6 connector. This connector, defined in CEM 5.1, replaces the 12VHPWR connector.
↑
Wallossek, Igor (3 July 2023).
"Rest in Peace 12VHPWR Connector - Welcome 12V-2×6 Connector, important modifications and PCIe Base 6
|
Exclusive"
.
igor'sLAB
. Retrieved
9 February
2025
.
↑
"12VHPWR and 12V-2x6 Compared"
.
Corsair
. Retrieved
9 February
2025
.
↑
Buildzoid (11 February 2025).
"How Nvidia made the 12VHPWR connector even worse"
.
Actually Hardcore Overclocking
. Retrieved
9 September
2025
–
via
YouTube
.
↑
Nilange, Manisha.
(T1S02 & T1S08) PCIe CEM Updates
. PCI-SIG Developers Conference 2023.
↑
"PCI-Express (PCIe) Add-in Card Connectors (Recommended)"
.
ATX Version 3 Multi Rail Desktop Platform Power Supply Design Guide, Revision 2.1a
.
Intel Corporation
. 1 November 2023. pp.
61–
65. Intel Document Number 336521-2.1a.
Archived
(PDF)
from the original on 14 February 2025
. Retrieved
9 September
2025
.
↑
PCI Express Card Electromechanical Specification Revision 5.1, Version 1.0
, 30 March 2023 – 10. PCI Express 48VHPWR Auxiliary Power Connector Definition
↑
"ATX Version 3 Multi Rail Desktop Platform Power Supply (2.1a, ATX 3.1)
::  PCIe* Add-in Card Power Excursions"
.
"ATX Version 3 Multi Rail Desktop Platform Power Supply (2.01, ATX 3.0)
::  PCIe* Add-in Card Power Excursions"
.
↑
"Understanding M.2, the interface that will speed up your next SSD"
. 8 February 2015.
↑
"MP1: Mini PCI Express / PCI Express Adapter"
.
hwtools.net
. 18 July 2014.
Archived
from the original on 3 October 2014
. Retrieved
28 September
2014
.
↑
IT Essentials Companion Guide v8
. Cisco Press. 9 July 2023.
ISBN
978-0-13-816625-0
.
↑
Mobile Computing Deployment and Management: Real World Skills for CompTIA Mobility+ Certification and Beyond
. John Wiley & Sons. 24 February 2015.
ISBN
978-1-118-82461-0
.
↑
"mSATA FAQ: A Basic Primer"
. Notebook review. Archived from the original on 12 February 2012.
↑
"Eee PC Research"
.
ivc
(wiki). Archived from the original on 30 March 2010
. Retrieved
26 October
2009
.
↑
"Desktop Board Solid-state drive (SSD) compatibility"
.
Intel
.
Archived
from the original on 2 January 2016.
↑
"How to distinguish the differences between M.2 cards | Dell US"
.
www.dell.com
. Retrieved
24 March
2020
.
↑
"PCI Express External Cabling 1.0 Specification"
.
Archived
from the original on 10 February 2007
. Retrieved
9 February
2007
.
↑
"PCI Express External Cabling Specification Completed by PCI-SIG"
. PCI SIG. 7 February 2007. Archived from
the original
on 26 November 2013
. Retrieved
7 December
2012
.
1
2
"OCuLink connectors and cables support new PCIe standard"
.
www.connectortips.com
. Archived from
the original
on 13 March 2017.
↑
Mokosiy, Vitaliy (9 October 2020).
"Untangling terms: M.2, NVMe, USB-C, SAS, PCIe, U.2, OCuLink"
.
Medium
. Retrieved
26 March
2021
.
↑
published, Anton Shilov (2 May 2024).
"PCI-SIG unveils CopprLink cables for PCIe 5 and 6 connections — PCIe 7.0 versions are under development"
.
Tom's Hardware
. Retrieved
16 April
2026
.
↑
"PCIe® Cabling – The Journey to CopprLink™ | PCI-SIG"
.
pcisig.com
. Retrieved
16 April
2026
.
↑
PCWorld (9 April 2026).
This eGPU embarrasses the rest
. Retrieved
16 April
2026
–
via YouTube.
↑
"Supermicro Universal I/O (UIO) Solutions"
. Supermicro.com.
Archived
from the original on 24 March 2014
. Retrieved
24 March
2014
.
↑
"Get ready for M-PCIe testing"
,
PC board design
, EDN
1
2
"PCI SIG discusses M-PCIe oculink & 4th gen PCIe"
,
The Register
,
UK
, 13 September 2013,
archived
from the original on 29 June 2017
↑
"PCI Express 4.0 Frequently Asked Questions"
.
pcisig.com
. PCI-SIG. Archived from
the original
on 18 May 2014
. Retrieved
18 May
2014
.
1
2
3
"PCI Express 3.0 Frequently Asked Questions"
.
pcisig.com
. PCI-SIG. Archived from
the original
on 1 February 2014
. Retrieved
1 May
2014
.
↑
"PCI-Express-Schnittstelle: PCIe 7.0 und die Basis für optische Verbindungen stehen, PCIe 8.0 folgt"
. 11 June 2025.
↑
"What does GT/s mean, anyway?"
.
TM World
.
Archived
from the original on 14 August 2012
. Retrieved
7 December
2012
.
↑
"Deliverable 12.2"
.
SE
: Eiscat. Archived from
the original
on 17 August 2010
. Retrieved
7 December
2012
.
↑
PCI SIG
,
archived
from the original on 6 July 2008
↑
"PCI Express Base 2.0 specification announced"
(PDF)
(Press release).
PCI-SIG
. 15 January 2007. Archived from
the original
(PDF)
on 4 March 2007
. Retrieved
9 February
2007
.
— note that in this press release the term
aggregate bandwidth
refers to the sum of incoming and outgoing bandwidth; using this terminology the aggregate bandwidth of full duplex 100BASE-TX is 200
Mbit/s.
↑
Smith, Tony (11 October 2006).
"PCI Express 2.0 final draft spec published"
.
The Register
.
Archived
from the original on 29 January 2007
. Retrieved
9 February
2007
.
↑
Key, Gary; Fink, Wesley (21 May 2007).
"Intel P35: Intel's Mainstream Chipset Grows Up"
.
AnandTech
. Archived from
the original
on 23 May 2007
. Retrieved
21 May
2007
.
↑
Huynh, Anh (8 February 2007).
"NVIDIA "MCP72" Details Unveiled"
.
AnandTech
. Archived from
the original
on 10 February 2007
. Retrieved
9 February
2007
.
↑
"Intel P35 Express Chipset Product Brief"
(PDF)
. Intel.
Archived
(PDF)
from the original on 26 September 2007
. Retrieved
5 September
2007
.
↑
Hachman, Mark (5 August 2009).
"PCI Express 3.0 Spec Pushed Out to 2010"
. PC Mag.
Archived
from the original on 7 January 2014
. Retrieved
7 December
2012
.
↑
"PCI Express 3.0 Bandwidth: 8.0 Gigatransfers/s"
. ExtremeTech. 9 August 2007.
Archived
from the original on 24 October 2007
. Retrieved
5 September
2007
.
↑
"PCI Special Interest Group Publishes PCI Express 3.0 Standard"
. X bit labs. 18 November 2010. Archived from
the original
on 21 November 2010
. Retrieved
18 November
2010
.
↑
"PCIe 3.1 and 4.0 Specifications Revealed"
.
eteknix.com
. July 2013.
Archived
from the original on 1 February 2016.
↑
"Trick or Treat... PCI Express 3.1 Released!"
.
synopsys.com
.
Archived
from the original on 23 March 2015.
↑
"PCI Express 4.0 evolution to 16
GT/s, twice the throughput of PCI Express 3.0 technology"
(press release). PCI-SIG. 29 November 2011. Archived from
the original
on 23 December 2012
. Retrieved
7 December
2012
.
↑
"Frequently Asked Questions
|
PCI-SIG"
.
pcisig.com
. Archived from
the original
on 20 October 2016.
↑
"PCIe 4.0 Heads to Fab, 5.0 to Lab"
.
EE Times
. 26 June 2016.
Archived
from the original on 28 August 2016
. Retrieved
27 August
2016
.
↑
"Mellanox Announces ConnectX-5, the Next Generation of 100G InfiniBand and Ethernet Smart Interconnect Adapter
|
NVIDIA"
.
www.mellanox.com
.
↑
"Mellanox Announces 200Gb/s HDR InfiniBand Solutions Enabling Record Levels of Performance and Scalability
|
NVIDIA"
.
www.mellanox.com
.
↑
"IDF: PCIe 4.0 läuft, PCIe 5.0 in Arbeit"
.
Heise Online
(in German). 18 August 2016.
Archived
from the original on 19 August 2016
. Retrieved
18 August
2016
.
↑
"Brian Thompto, POWER9 Processor for the Cognitive Era"
(PDF)
.
↑
2016 IEEE Hot Chips 28 Symposium (HCS), 21–23 Aug. 2016
1
2
Born, Eric (8 June 2017).
"PCIe 4.0 specification finally out with 16 GT/s on tap"
. Tech Report.
Archived
from the original on 8 June 2017
. Retrieved
8 June
2017
.
↑
"IBM Unveils Most Advanced Server for AI"
.
www-03.ibm.com
. 5 December 2017. Archived from
the original
on 8 December 2017.
↑
"IBM Power System AC922 (8335-GTG) server helps you to harness breakthrough accelerated AI, HPDA, and HPC performance for faster time to insight"
.
www.ibm.com
. 5 December 2017.
Archived
from the original on 27 June 2024
. Retrieved
25 April
2025
.
↑
"NETINT Introduces Codensity with Support for PCIe 4.0 – NETINT Technologies"
.
NETINT Technologies
. 17 July 2018. Archived from
the original
on 29 September 2018
. Retrieved
28 September
2018
.
↑
Mujtaba, Hassan (9 January 2019).
"AMD Ryzen 3000 Series CPUs Based on Zen 2 Launching in Mid of 2019"
.
↑
Alcorn, Paul (3 June 2019).
"AMD Nixes PCIe 4.0 Support on Older Socket AM4 Motherboards, Here's Why"
.
Tom's Hardware
. Retrieved
10 June
2019
.
`{{cite web}}`
:  CS1 maint: deprecated archival service (
link
)
↑
Alcorn, Paul (10 January 2019).
"PCIe 4.0 May Come to all AMD Socket AM4 Motherboards (Updated)"
.
Tom's Hardware
. Retrieved
10 June
2019
.
`{{cite web}}`
:  CS1 maint: deprecated archival service (
link
)
↑
Cutress, Dr. Ian (13 August 2020).
"Tiger Lake IO and Power"
.
Anandtech
. Archived from
the original
on 2 October 2020.
↑
"1,2,3,4,5... It's Official, PCIe 5.0 is Announced | synopsys.com"
.
www.synopsys.com
. Retrieved
7 June
2017
.
↑
"PLDA Announces Availability of XpressRICH5 PCIe 5.0 Controller IP | PLDA.com"
.
www.plda.com
. Retrieved
28 June
2018
.
↑
"XpressRICH5 for ASIC | PLDA.com"
.
www.plda.com
. Retrieved
28 June
2018
.
↑
"Doubling Bandwidth in Under Two Years: PCI Express® Base Specification Revision 5.0, Version 0.9 is Now Available to Members"
.
pcisig.com
. Retrieved
12 December
2018
.
↑
"PCIe 5.0 Is Ready For Prime Time"
.
tomshardware.com
. 17 January 2019
. Retrieved
18 January
2019
.
↑
"PCI-SIG Achieves 32GT/s with New PCI Express 5.0 Specification"
.
www.businesswire.com
. 29 May 2019.
↑
"PCI-Express 5.0: China stellt ersten Controller vor"
.
PC Games Hardware
. 18 November 2019.
↑
"IBM's POWER10 Processor, Hot Chips 32, August 16–18, 2020"
(PDF)
.
↑
Power E1080 Enterprise server delivers a uniquely architected platform to help securely and efficiently scale core operational and AI applications in a hybrid cloud, IBM Europe Hardware Announcement ZG21-0059
1
2
IBM Power E1080 Technical Overview and Introduction
↑
"Intel Unveils 12th Gen Intel Core, Launches World's Best Gaming"
. Intel.com. 31 December 2021
. Retrieved
16 February
2022
.
↑
"NVIDIA Announces Hopper Architecture, the Next Generation of Accelerated Computing"
.
↑
"AMD Showcases Industry-Leading Gaming, Commercial, and Mainstream PC Technologies at COMPUTEX 2022"
. AMD.com
. Retrieved
23 May
2022
.
↑
"4th Gen AMD EPYC Processor Architecture"
. AMD.com
. Retrieved
12 November
2022
.
1
2
"PCI-SIG® Announces Upcoming PCI Express® 6.0 Specification to Reach 64 GT/s"
.
www.businesswire.com
. 18 June 2019.
↑
Smith, Ryan.
"PCI Express Bandwidth to Be Doubled Again: PCIe 6.0 Announced, Spec to Land in 2021"
.
www.anandtech.com
. Archived from
the original
on 18 June 2019.
↑
"PCI Express 6.0 Reaches Version 0.5 Ahead Of Finalization Next Year – Phoronix"
.
www.phoronix.com
.
↑
Shilov, Anton (4 November 2020).
"PCIe 6.0 Specification Hits Milestone: Complete Draft Is Ready"
.
Tom's Hardware
.
↑
Yanes, Al.
"PCIe 6.0 Specification, Version 0.9: One Step Closer to Final Release | PCI-SIG"
.
pcisig.com
. Retrieved
6 October
2021
.
↑
"PCI-SIG Releases PCIe 6.0 Specification Delivering Record Performance to Power Big Data Applications"
. Business Wire. 11 January 2022
. Retrieved
16 February
2022
.
↑
"The Evolution of the PCI Express Specification: On its Sixth Generation, Third Decade and Still Going Strong"
. Pci-Sig. 11 January 2022
. Retrieved
16 February
2022
.
↑
Debendra Das Sharma (8 June 2020).
"PCIe 6.0 Specification: The Interconnect for I/O Needs of the Future"
. PCI-SIG. p.
8. Archived from
the original
on 30 October 2021.
↑
"Pushing the Envelope with PCIe 6.0: Bringing PAM4 to PCIe"
(PDF)
. Retrieved
16 February
2022
.
↑
"PowerPoint Presentation"
(PDF)
. Retrieved
16 February
2022
.
↑
Hachman, Mark (1 August 2025).
"The world's first bleeding-edge PCIe 6.0 SSD is here (but not for you)"
.
PCWorld
. Retrieved
16 September
2025
.
↑
Hachman, Mark (11 June 2025).
"PCIe 6.0 devices poised for 2025 launch, ushering in next-gen connectivity"
.
PCWorld
. Retrieved
16 September
2025
.
1
2
Hachman, Mark (19 March 2025).
"PCI Express 7.0's blazing speeds are nearly here, but PCIe 6 is still vapor"
.
PCWorld
. Retrieved
1 July
2025
.
↑
"PCI-SIG Announces PCI Express 7.0 Specification to Reach 128 GT/s"
. Business Wire. 21 June 2022
. Retrieved
25 June
2022
.
↑
"PCIe 7.0 Specification, Version 0.5 Now Available: Full Draft Available to Members"
.
pcisig.com
. Retrieved
3 April
2024
.
↑
"Progressing on Track: PCIe 7.0 Specification, Version 0.7 Now Available for Member Review"
.
pcisig.com
. Retrieved
17 January
2025
.
↑
"PCIe 7.0 Specification, Version 0.9: Final Draft Now Available for Member Review"
.
pcisig.com
. Retrieved
19 March
2025
.
↑
"PCI-SIG Releases PCIe 7.0 Specification to Support the Bandwidth Demands of Artificial Intelligence at 128.0 GT/s Transfer Rates"
. Business Wire. 11 June 2025
. Retrieved
11 June
2025
.
↑
"PCI-SIG Announces PCIe 8.0 Specification Targeted for Release by 2028 | PCI-SIG"
.
pcisig.com
. Retrieved
6 August
2025
.
↑
"PLX demo shows PCIe over fiber as data center clustering interconnect"
.
Cabling install
. Penn Well
. Retrieved
29 August
2012
.
[
permanent dead link
]
↑
"Introduced second generation PCI Express Gen 2 over fiber optic systems"
. Adnaco. 22 April 2011.
Archived
from the original on 4 October 2012
. Retrieved
29 August
2012
.
↑
"PCIe Active Optical Cable System"
.
Archived
from the original on 30 December 2014
. Retrieved
23 October
2015
.
↑
"IBM Power Systems E870 and E880 Technical Overview and Introduction"
(PDF)
.
↑
"Acer, Asus to Bring Intel's Thunderbolt Speed Technology to Windows PCs"
.
PC World
. 14 September 2011.
Archived
from the original on 18 January 2012
. Retrieved
7 December
2012
.
↑
Kevin Parrish (28 June 2013).
"PCIe for Mobile Launched; PCIe 3.1, 4.0 Specs Revealed"
.
Tom's Hardware
. Retrieved
10 July
2014
.
↑
"PCI Express 4.0 Draft 0.7 & PIPE 4.4 Specifications – What Do They Mean to Designers? — Synopsys Technical Article | ChipEstimate.com"
.
www.chipestimate.com
. Retrieved
28 June
2018
.
↑
"PCI Express 1×, 4×, 8×, 16×  bus pinout and wiring @"
.
RU
: Pinouts.
Archived
from the original on 25 November 2009
. Retrieved
7 December
2009
.
↑
"PHY Interface for the PCI Express Architecture"
(PDF)
(version 2.00
ed.). Intel. Archived from
the original
(PDF)
on 17 March 2008
. Retrieved
21 May
2008
.
↑
"PCI Express System Architecture"
(PDF)
.
↑
"Communications"
.
Intel
.
↑
"Mechanical Drawing for PCI Express Connector"
. Interface bus
. Retrieved
7 December
2007
.
↑
"FCi schematic for PCIe connectors"
(PDF)
. FCI connect. Archived from
the original
(PDF)
on 20 September 2008
. Retrieved
7 December
2007
.
↑
"Reducing Interrupt Latency Through the Use of Message Signaled Interrupts"
(PDF)
.
↑
PCI Express Base Specification, Revision 3.0
Table 4-24
↑
"PCIe Data Transmission Overview"
(PDF)
.
Microchip Technology
.
↑
CompTIA A+ Exam Cram (Exams 220-602, 220-603, 220-604)
. Pearson Education. 19 July 2007.
ISBN
978-0-7686-9003-3
.
↑
Computer Peripherals And Interfaces
. Technical Publications Pune. 2008.
ISBN
9788184313086
.
Archived
from the original on 25 February 2014
. Retrieved
23 July
2009
.
1
2
3
4
5
Lawley, Jason (28 October 2014).
"Understanding Performance of PCI Express Systems"
(PDF)
. 1.2. Xilinx.
↑
"NVIDIA Introduces NVIDIA Quadro Plex – A Quantum Leap in Visual Computing"
.
Nvidia
. 1 August 2006.
Archived
from the original on 24 August 2006
. Retrieved
14 July
2018
.
↑
"Quadro Plex VCS – Advanced visualization and remote graphics"
. nVidia.
Archived
from the original on 28 April 2011
. Retrieved
11 September
2010
.
↑
"XGP"
.
ATI
. AMD. Archived from
the original
on 29 January 2010
. Retrieved
11 September
2010
.
↑
Fujitsu-Siemens Amilo GraphicBooster External Laptop GPU Released
, 3 December 2008,
archived
from the original on 16 October 2015
, retrieved
9 August
2015
↑
DynaVivid Graphics Dock from Acer arrives in France, what about the US?
, 11 August 2010,
archived
from the original on 16 October 2015
, retrieved
9 August
2015
↑
Dougherty, Steve (22 May 2010),
"MSI to showcase 'GUS' external graphics solution for laptops at Computex"
,
TweakTown
↑
Hellstrom, Jerry (9 August 2011),
"ExpressCard trying to pull a (not so) fast one?"
,
PC Perspective
(editorial),
archived
from the original on 1 February 2016
↑
"PE4H V3.2 (PCIe x16 Adapter)"
. Hwtools.net.
Archived
from the original on 14 February 2014
. Retrieved
5 February
2014
.
↑
O'Brien, Kevin (8 September 2010),
"How to Upgrade Your Notebook Graphics Card Using DIY ViDOCK"
,
Notebook review
, archived from the original on 13 December 2013
↑
Lal Shimpi, Anand (7 September 2011),
"The Thunderbolt Devices Trickle In: Magma's ExpressBox 3T"
,
AnandTech
, archived from
the original
on 4 March 2016
↑
"MSI GUS II external GPU enclosure with Thunderbolt"
.
The Verge
(hands-on). 10 January 2012.
Archived
from the original on 13 February 2012
. Retrieved
12 February
2012
.
↑
"PCI express graphics, Thunderbolt"
,
Tom's hardware
, 17 September 2012
↑
"M logics M link Thunderbold chassis no shipping"
,
Engadget
, 13 December 2012,
archived
from the original on 25 June 2017
↑
Burns, Chris (17 October 2017),
"2017 Razer Blade Stealth and Core V2 detailed"
,
SlashGear
,
archived
from the original on 17 October 2017
↑
"CompactFlash Association readies next-gen XQD format, promises write speeds of 125
MB/s and up"
. Engadget. 8 December 2011.
Archived
from the original on 19 May 2014
. Retrieved
18 May
2014
.
↑
Zsolt Kerekes (December 2011).
"What's so very different about the design of Fusion-io's ioDrives / PCIe SSDs?"
. storagesearch.com.
Archived
from the original on 23 September 2013
. Retrieved
2 October
2013
.
↑
"Fusion-io ioDrive Duo Enterprise PCIe Review"
. storagereview.com. 16 July 2012. Archived from
the original
on 4 October 2013
. Retrieved
2 October
2013
.
↑
"OCZ Demos 4
TiB, 16
TiB Solid-State Drives for Enterprise"
. X-bit labs. Archived from
the original
on 25 March 2013
. Retrieved
7 December
2012
.
↑
"Enabling Higher Speed Storage Applications with SATA Express"
. SATA-IO.
Archived
from the original on 27 November 2012
. Retrieved
7 December
2012
.
↑
"SATA M.2 Card"
. SATA-IO.
Archived
from the original on 3 October 2013
. Retrieved
14 September
2013
.
↑
"SCSI Express"
. SCSI Trade Association. Archived from
the original
on 27 January 2013
. Retrieved
27 December
2012
.
↑
Meduri, Vijay (24 January 2011).
"A Case for PCI Express as a High-Performance Cluster Interconnect"
. HPCwire.
Archived
from the original on 14 January 2013
. Retrieved
7 December
2012
.
↑
Evan Koblentz (3 February 2017).
"New PCI Express 4.0 delay may empower next-gen alternatives"
.
Tech Republic
.
Archived
from the original on 1 April 2017
. Retrieved
31 March
2017
.
↑
Cutress, Ian.
"CXL Specification 1.0 Released: New Industry High-Speed Interconnect From Intel"
.
www.anandtech.com
. Archived from
the original
on 11 March 2019
. Retrieved
9 August
2019
.
↑
"Integrators List | PCI-SIG"
.
pcisig.com
. Retrieved
27 March
2019
.
Further reading
[
edit
]
Budruk, Ravi; Anderson, Don; Shanley, Tom (2003), Winkles, Joseph 'Joe' (ed.),
PCI Express System Architecture
, Mind share PC system architecture, Addison-Wesley,
ISBN
978-0-321-15630-3
, 1120 pp.
Solari, Edward; Congdon, Brad (2003),
Complete PCI Express Reference: Design Implications for Hardware and Software Developers
, Intel,
ISBN
978-0-9717861-9-6
, 1056 pp.
Wilen, Adam; Schade, Justin P; Thornburg, Ron (April 2003),
Introduction to PCI Express: A Hardware and Software Developer's Guide
, Intel,
ISBN
978-0-9702846-9-3
, 325 pp.
External links
[
edit
]
Media related to
PCIe
at Wikimedia Commons
PCI-SIG Specifications
v
t
e
Technical
and
de facto
standards
for
wired
computer buses
General
System bus
Front-side bus
Back-side bus
Daisy chain
Control bus
Address bus
Bus contention
Bus mastering
Network on a chip
Plug and play
Double data rate
Quad data rate
List of bus bandwidths
Standards
SS-50 bus
S-100 bus
Multibus
Unibus
VAXBI
MBus
STD Bus
SMBus
Q-Bus
Europe Card Bus
ISA
STEbus
Zorro II
Zorro III
CAMAC
FASTBUS
LPC
HP Precision Bus
EISA
VME
VXI
VXS
VPX
NuBus
TURBOchannel
MCA
SBus
VLB
HP GSC bus
InfiniBand
Ethernet
UPA
PCI
PCI Extended (PCI-X)
PXI
PCI Express (PCIe)
AGP
Compute Express Link (CXL)
Direct Media Interface (DMI)
RapidIO
Intel QuickPath Interconnect
NVLink
HyperTransport
Infinity Fabric
Intel Ultra Path Interconnect
Coherent Accelerator Processor Interface (CAPI)
SpaceWire
Storage
ST-506
ESDI
SDI
IPI
SMD
Floppy connector
Parallel ATA (PATA)
Bus and Tag
DSSI
HIPPI
Serial ATA (SATA)
SCSI
Parallel
SAS
ESCON
Fibre Channel
SSA
SATAe
PCI Express (via
AHCI
or
NVMe
logical device interface)
Peripheral
Apple Desktop Bus
Atari SIO
DCB
Commodore bus
HP-IL
HIL
MIDI
RS-232
RS-422
RS-423
RS-485
Lightning
DMX512-A
IEEE-488 (GPIB)
IEEE-1284 (parallel port)
IEEE-1394 (FireWire)
UNI/O
1-Wire
I
2
C
(
ACCESS.bus
,
PMBus
,
SMBus
)
I3C
SPI
D²B
Parallel SCSI
Profibus
USB
Camera Link
External PCIe
Thunderbolt
CAN bus
Audio
ADAT Lightpipe
AES3
Intel HD Audio
I
2
S
MADI
McASP
S/PDIF
TOSLINK
Portable
PC Card
ExpressCard
Embedded
Multidrop bus
CoreConnect
AMBA
(
AXI
)
Wishbone
SLIMbus
Interfaces are listed by their speed in the (roughly) ascending order, so the interface at the end of each section should be the fastest.
Category
v
t
e
Basic
computer
components
Input devices
Pointing devices
Graphics tablet
Game controller
Light pen
Mouse
Optical
Optical trackpad
Pointing stick
Touchpad
Touchscreen
Trackball
Other
Keyboard
Image scanner
Graphics card
GPU
Microphone
Refreshable braille display
Sound card
Sound chip
Webcam
Softcam
Output devices
Monitor
Screen
Refreshable braille display
Printer
Plotter
Speakers
Sound card
Graphics card
Removable
data storage
Disk pack
Floppy disk
Optical disc
CD
DVD
Blu-ray
Flash memory
Memory card
USB flash drive
Computer case
Central processing unit
Microprocessor
Motherboard
Memory
RAM
BIOS
Data storage
HDD
SSD
(
SATA
/
NVMe
)
SSHD
Power supply
SMPS
MOSFET
Power MOSFET
VRM
Network interface controller
Fax modem
Expansion card
Ports
Current
Ethernet
USB
Thunderbolt
Analog audio jack
DisplayPort
HDMI
Obsolete
FireWire
(IEEE 1394)
Parallel port
Serial port
Game port
PS/2 port
eSATA
DVI
VGA
Related
History of computing hardware
History of computing hardware (1960s–present)
List of pioneers in computer science
Retrieved from "
https://en.wikipedia.org/w/index.php?title=PCI_Express&oldid=1378651517
"
Categories
:
Computer-related introductions in 2004
Peripheral Component Interconnect
Serial buses
Computer standards
Motherboard expansion slot
Hidden categories:
Articles with short description
Short description matches Wikidata
Use dmy dates from October 2020
Wikipedia articles needing clarification from January 2025
Articles needing additional references from March 2018
All articles needing additional references
All articles with unsourced statements
Articles with unsourced statements from July 2022
Articles with excerpts
All articles with vague or ambiguous time
Vague or ambiguous time from September 2025
Articles containing potentially dated statements from 2015
All articles containing potentially dated statements
Articles with unsourced statements from July 2019
Articles with unsourced statements from September 2021
Articles containing potentially dated statements from 2012
Articles containing potentially dated statements from 2013
All articles that may have off-topic sections
Wikipedia articles that may have off-topic sections from September 2021
Articles with unsourced statements from February 2021
Wikipedia articles needing clarification from September 2024
CS1: unfit URL
CS1 German-language sources (de)
All articles with dead external links
Articles with dead external links from April 2025
Articles with permanently dead external links
CS1 maint: deprecated archival service
Commons category link from Wikidata
Search
Search
PCI Express
42 languages
Add topic